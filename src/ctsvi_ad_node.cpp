// ctsvi_ad_node.cpp
// Pinocchio + CppAD 自动微分离散拉格朗日积分器（修正版）

// include order matters: CppAD -> Eigen -> Pinocchio
#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>
// fix Eigen <-> CppAD compatibility for isfinite()
namespace Eigen {
    namespace numext {
        template<>
        EIGEN_STRONG_INLINE bool isfinite(const CppAD::AD<double>&) {
            return true;
        }
    }
}

#include <Eigen/Dense>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <rclcpp/rclcpp.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <numeric>
#include <string>
#include <sstream>
#include <algorithm>
#include <regex>
#include <iomanip>

using namespace std::chrono;
using namespace pinocchio;

// ---------- 类型别名 ----------
using ADScalar = CppAD::AD<double>;
using VecAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;
using MatAD = Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vec  = Eigen::VectorXd;
using Mat  = Eigen::MatrixXd;
using ModelAD = pinocchio::ModelTpl<ADScalar>;
using DataAD  = pinocchio::DataTpl<ADScalar>;

// 计算末端位姿（位置 + 旋转矩阵）
std::pair<Eigen::Vector3d, Eigen::Matrix3d>
compute_end_effector_pose(const Model &model,
                          Data &data,
                          int frame_id,
                          const Vec &q)
{
    // Update joint FK
    pinocchio::forwardKinematics(model, data, q);

    // Update placements of all joints and frames
    // pinocchio::updateGlobalPlacements(model, data);
    pinocchio::updateFramePlacements(model, data);

    Eigen::Vector3d ee_pos = data.oMf[frame_id].translation();
    Eigen::Matrix3d ee_rot = data.oMf[frame_id].rotation();

    return {ee_pos, ee_rot};
}

// ---------- double 版本辅助（用于能量输出等） ----------
Mat inertia_matrix(const Model &model, Data &data, const Vec &q)
{
    pinocchio::crba(model, data, q);
    return data.M;
}

// Kinetic energy using midpoint q_mid and qdot
double kinetic_energy(const Model &model, Data &data, const Vec &q_mid, const Vec &qdot)
{
    // compute inertia at q_mid
    pinocchio::crba(model, data, q_mid);
    Mat M = data.M;
    return 0.5 * qdot.transpose() * M * qdot;
}

double potential_energy(const Model &model, Data &data, const Vec &q)
{
    return pinocchio::computePotentialEnergy(model, data, q);
}

// ------------------ 离散拉格朗日函数（AD 版） ------------------
ADScalar discreteLagrangian(const ModelAD &model, DataAD &data,
                            const VecAD &q0, const VecAD &q1, double h)
{
    VecAD q_mid = 0.5 * (q0 + q1);
    VecAD dq = (q1 - q0) / h;

    // crba on AD model fills data.M of ADScalar
    pinocchio::crba(model, data, q_mid);
    MatAD M = data.M; // already AD type

    // kinetic energy (AD)
    ADScalar T = ADScalar(0.5) * dq.transpose() * (M * dq);

    // potential energy: use Pinocchio computePotentialEnergy templated
    ADScalar U = pinocchio::computePotentialEnergy(model, data, q_mid);

    return ADScalar(h) * (T - U);
}

// ------------------ D1 Ld 自动微分 ------------------
Vec D_1(const ModelAD &model_ad, DataAD &data_ad,
         const Vec &q0, const Vec &q1, double h)
{
    const int n = q0.size();

    // 1) prepare CppAD independent vector (CppAD::vector<ADScalar>)
    CppAD::vector<ADScalar> x_ad(n);
    for (int i = 0; i < n; ++i) x_ad[i] = q0[i];

    CppAD::Independent(x_ad);

    // 2) map to Eigen VecAD for calling discreteLagrangian
    VecAD q0_ad = Eigen::Map<VecAD>(x_ad.data(), n);
    VecAD q1_ad = q1.cast<ADScalar>();

    // 3) compute Ld (AD)
    ADScalar Ld = discreteLagrangian(model_ad, data_ad, q0_ad, q1_ad, h);

    // 4) create dependent vector y (CppAD::vector<ADScalar>)
    CppAD::vector<ADScalar> y(1);
    y[0] = Ld;

    // 5) create function f : x -> y
    CppAD::ADFun<double> f;
    f.Dependent(x_ad, y);

    // 6) evaluate Jacobian at double x = q0
    CppAD::vector<double> x0(n);
    for (int i = 0; i < n; ++i) x0[i] = q0[i];

    CppAD::vector<double> jac = f.Jacobian(x0); // length n (since y dim = 1)

    Vec grad(n);
    for (int i = 0; i < n; ++i) grad[i] = jac[i];
    return grad;
}

// ------------------ D2 Ld 自动微分 ------------------
Vec D_2(const ModelAD &model_ad, DataAD &data_ad,
         const Vec &q0, const Vec &q1, double h)
{
    const int n = q1.size();

    // independent variables are q1 now
    CppAD::vector<ADScalar> x_ad(n);
    for (int i = 0; i < n; ++i) x_ad[i] = q1[i];

    CppAD::Independent(x_ad);

    VecAD q0_ad = q0.cast<ADScalar>();
    VecAD q1_ad = Eigen::Map<VecAD>(x_ad.data(), n);

    ADScalar Ld = discreteLagrangian(model_ad, data_ad, q0_ad, q1_ad, h);

    CppAD::vector<ADScalar> y(1);
    y[0] = Ld;

    CppAD::ADFun<double> f;
    f.Dependent(x_ad, y);

    CppAD::vector<double> x0(n);
    for (int i = 0; i < n; ++i) x0[i] = q1[i];

    CppAD::vector<double> jac = f.Jacobian(x0);

    Vec grad(n);
    for (int i = 0; i < n; ++i) grad[i] = jac[i];
    return grad;
}

// ---------- 变分积分器牛顿法 ----------
struct SolverInfo { bool converged; std::string reason; int iterations; double residual_norm; };

std::pair<Vec, SolverInfo> solve_q_next(const ModelAD &model_ad,
                                        DataAD &data_ad,
                                        const Vec &q_prev,
                                        const Vec &q_curr,
                                        const Vec &tau_k,
                                        double h,
                                        int max_iters=50,
                                        double tol=1e-8)
{
    int n = q_curr.size();
    Vec q_next = q_curr;
    SolverInfo info{false, "", 0, 0.0};

    for (int it = 0; it < max_iters; ++it)
    {
        Vec D2 = D_2(model_ad, data_ad, q_prev, q_curr, h);
        Vec D1 = D_1(model_ad, data_ad, q_curr, q_next, h);
        Vec R = D2 + D1 - h * tau_k;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q_next, info}; }

        // numeric Jacobian of D1 wrt q_next (could be replaced by AD Jacobian)
        Mat J = Mat::Zero(n, n);
        double eps = 1e-6;
        for (int j = 0; j < n; ++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps;
            Vec D1p = D_1(model_ad, data_ad, q_curr, q_next + dq, h);
            Vec D1m = D_1(model_ad, data_ad, q_curr, q_next - dq, h);
            J.col(j) = (D1p - D1m) / (2.0 * eps);
        }

        Mat A = J + 1e-9 * Mat::Identity(n, n);
        Eigen::ColPivHouseholderQR<Mat> solver(A);
        if (solver.rank() < n)
        {
            info.converged = false;
            info.reason = "singular_jacobian";
            return {q_next, info};
        }
        Vec delta = solver.solve(-R);
        q_next += delta;
    }

    info.converged = false;
    info.reason = "max_iters";
    return {q_next, info};
}

// ---------- CSV 输出 ----------
void write_csv(const std::string &filename, const std::vector<Vec> &rows)
{
    if (rows.empty()) return;
    std::ofstream ofs(filename);
    int ncols = rows[0].size();
    for (size_t r = 0; r < rows.size(); ++r)
    {
        for (int c = 0; c < ncols; ++c)
        {
            ofs << std::setprecision(15) << rows[r](c);
            if (c + 1 < ncols) ofs << ",";
        }
        ofs << "\n";
    }
}

void write_csv_scalar_series(const std::string &filename, const std::vector<double> &rows)
{
    std::ofstream ofs(filename);
    for (double v : rows) ofs << std::setprecision(15) << v << "\n";
}

void write_csv_3d(const std::string &filename,
                  const std::vector<Eigen::Vector3d> &rows)
{
    std::ofstream ofs(filename);
    for (const auto &v : rows)
    {
        ofs << std::setprecision(15)
            << v(0) << "," << v(1) << "," << v(2) << "\n";
    }
}

std::string expand_user(const std::string &path)
{
    if (!path.empty() && path[0] == '~') {
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

std::string fmt_double_label(double v)
{
    std::ostringstream oss;
    oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
    oss<<std::fixed<<std::setprecision(2)<<v;
    std::string s = oss.str();
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

std::pair<double,double> read_lyap_from_config()
{
    std::string cfg = "src/vi_2p/config/vi_params.yaml";
    std::ifstream ifs(cfg);
    double a = 0.0, b = 0.0;
    if (!ifs) return {a,b};
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::smatch m;
    try {
        std::regex ra("lyap_alpha\\s*:\\s*([0-9.eE+-]+)");
        std::regex rb("lyap_beta\\s*:\\s*([0-9.eE+-]+)");
        if (std::regex_search(content, m, ra)) a = std::stod(m[1].str());
        if (std::regex_search(content, m, rb)) b = std::stod(m[1].str());
    } catch (...) {}
    return {a,b};
}

// ---------- Main ----------
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "ctsvi_ad_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // 获取参数值
    std::vector<double> q_init_vec;
    double q_init_scalar = 0.2;
    bool q_init_is_vec = node->get_parameter("q_init", q_init_vec);
    if (!q_init_is_vec) node->get_parameter("q_init", q_init_scalar);
    double timestep = node->get_parameter("timestep").as_double();
    double duration = node->get_parameter("duration").as_double();
    // std::string urdf_path = node->get_parameter("urdf_path").as_string();
    std::string urdf_path = expand_user(node->get_parameter("urdf_path").as_string());

    RCLCPP_INFO(node->get_logger(), "Using URDF: %s", urdf_path.c_str());
    // print q_init (scalar or vector)
    std::string q_init_print;
    if (q_init_is_vec) {
        std::ostringstream qqs; qqs << "[";
        for (size_t i = 0; i < q_init_vec.size(); ++i) {
            if (i) qqs << ", ";
            qqs << std::setprecision(6) << q_init_vec[i];
        }
        qqs << "]";
        q_init_print = qqs.str();
        RCLCPP_INFO(node->get_logger(), "Initial = %s, Duration = %.1f s, Timestep = %.3f s", q_init_print.c_str(), duration, timestep);
    } else {
        RCLCPP_INFO(node->get_logger(), "Initial = %.6f rad, Duration = %.1f s, Timestep = %.3f s", q_init_scalar, duration, timestep);
    }

    // build double model and data (for non-AD tasks like energy logging)
    Model model;
    try {
        pinocchio::urdf::buildModel(urdf_path, model);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Error loading URDF: %s", e.what());
        return 1;
    }
    Data data(model);
    model.gravity.linear(Eigen::Vector3d(0,0,-9.81));

    // 获取末端 frame id
    int link_tcp_id = model.getFrameId("link_tcp");
    RCLCPP_INFO(node->get_logger(), "link_tcp_id=%d:", link_tcp_id);
    if (link_tcp_id == -1) {
        RCLCPP_ERROR(node->get_logger(), "TCP not found!");
        return 1;
    }

    // Print joints
    // std::cout << "Model has nq=" << model.nq << " nv=" << model.nv << " joints: \n";
    RCLCPP_INFO(node->get_logger(), "Model has nq=%d nv=%d joints:", model.nq, model.nv);

    // cast to AD model/data
    ModelAD model_ad = model.cast<ADScalar>();
    DataAD data_ad(model_ad);

    int n = model.nq;
    int n_steps = static_cast<int>(duration / timestep);

    Vec q_prev;
    if (q_init_is_vec) {
        q_prev = Vec::Zero(n);
        for (int i = 0; i < n; ++i) {
            if (i < static_cast<int>(q_init_vec.size())) q_prev(i) = q_init_vec[i];
            else q_prev(i) = q_init_vec.back();
        }
    } else {
        q_prev = Vec::Constant(n, q_init_scalar);
    }
    Vec v_prev = Vec::Zero(n);
    Vec tau_k = Vec::Zero(n);

    std::vector<Vec> q_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> delta_energy_history;
    std::vector<double> energy_T_history;
    std::vector<double> energy_U_history;
    std::vector<Eigen::Vector3d> ee_history;
    std::vector<Vec> momentum_history;

    q_history.push_back(q_prev);

    auto t_start = high_resolution_clock::now();

    // initial energy
    // Mat M0 = inertia_matrix(model, data, q_prev);
    // double T0 = 0.5 * v_prev.transpose() * M0 * v_prev;
    // double U0 = potential_energy(model, data, q_prev);
    // energy_history.push_back(T0 + U0);

    // 初始时刻
    Mat M0 = inertia_matrix(model, data, q_prev);
    Vec qdot0 = v_prev;
    double T0 = 0.5 * qdot0.transpose() * M0 * qdot0;
    double U0 = potential_energy(model, data, q_prev);
    double total_energy = T0 + U0;
    energy_T_history.push_back(T0);
    energy_U_history.push_back(U0);
    energy_history.push_back(total_energy);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    auto [ee_pos0, ee_rot0] = compute_end_effector_pose(model, data, link_tcp_id, q_prev);
    ee_history.push_back(ee_pos0);

    // initial VI step
    auto [q_curr, info_init] = solve_q_next(model_ad, data_ad, q_prev, q_prev + v_prev * timestep, tau_k, timestep);
    q_history.push_back(q_curr);

    double T = kinetic_energy(model, data, (q_curr + q_prev) / 2.0, (q_curr - q_prev)/timestep);
    double U = potential_energy(model, data, (q_curr + q_prev) / 2.0);
    energy_T_history.push_back(T);
    energy_U_history.push_back(U);
    energy_history.push_back(T+U);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    auto [ee_pos1, _ee_rot1] = compute_end_effector_pose(model, data, link_tcp_id, q_curr);
    ee_history.push_back(ee_pos1);

    std::vector<double> runtimes;

    int bar_width = 50;
    double avg_time = 0.0;
    double time_left = 0.0;

    for (int step = 0; step < n_steps - 1; ++step)
    {
        auto t0 = high_resolution_clock::now();
        auto [q_next, info] = solve_q_next(model_ad, data_ad, q_history[q_history.size()-2],
                                           q_history[q_history.size()-1], tau_k, timestep);
        q_history.push_back(q_next);
        time_history.push_back(step * timestep);

        // Vec qdot = (q_next - q_history[q_history.size()-2]) / timestep;
        // Mat M = inertia_matrix(model, data, q_next);
        // double T = 0.5 * qdot.transpose() * M * qdot;
        // double U = potential_energy(model, data, q_next);
        // energy_history.push_back(T + U);

        T = kinetic_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0, (q_history.back() - q_history[q_history.size()-2])/timestep);
        U = potential_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0);
        energy_T_history.push_back(T);
        energy_U_history.push_back(U);
        energy_history.push_back(T+U);
        delta_energy_history.push_back(energy_history.back() - energy_history.front());

        auto [ee_pos, ee_rot] = compute_end_effector_pose(model, data, link_tcp_id, q_next);
        ee_history.push_back(ee_pos);

        // compute generalized momentum p = M(q_mid) * qdot
        Vec qdot = (q_next - q_history[q_history.size()-2]) / timestep;
        Vec qmid = 0.5 * (q_next + q_history[q_history.size()-2]);
        Mat Mmid = inertia_matrix(model, data, qmid);
        Vec p = Mmid * qdot;
        momentum_history.push_back(p);

        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        // 计算完成比例
        double progress = double(step + 1) / n_steps;
        int pos = int(bar_width * progress);

        if (!runtimes.empty()) {
            avg_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
        }
        time_left = avg_time * (n_steps - step);

        // 构建进度条字符串
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%, " << int(time_left / 60) << "mins " << int(time_left) % 60 << "s left...";
        std::cout.flush();  // 强制输出
    }
    std::cout << "\n";
    time_history.push_back((n_steps-1) * timestep);
    time_history.push_back(n_steps * timestep);

    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    if (!runtimes.empty()) {
        avg_time = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
    }

    // std::cout << "Simulation finished, wall time: " << total_elapsed << " s\n";
    // std::cout << "Average step time: " << avg_time*1e6 << " microseconds\n";
    RCLCPP_INFO(node->get_logger(),
        "Simulation finished, wall time: %f s, Average step time: %f ms",
        total_elapsed,
        avg_time*1e3);


    // Save into parameterized folder including lyap params: src/vi_2p/csv/q<q>_dt<dt>_T<T>_a<alpha>_b<beta>/ctsvi_ad/
    auto [a_val, b_val] = read_lyap_from_config();
    std::string q_label;
    if (q_init_is_vec) {
        std::ostringstream qqs; for (size_t i = 0; i < q_init_vec.size(); ++i) { if (i) qqs << "_"; qqs << fmt_double_label(q_init_vec[i]); }
        q_label = qqs.str();
    } else {
        q_label = fmt_double_label(q_init_scalar);
    }
    std::string params_label = std::string("q") + q_label
        + std::string("_dt") + fmt_double_label(timestep)
        + std::string("_T") + fmt_double_label(duration)
        + std::string("_a") + fmt_double_label(a_val)
        + std::string("_b") + fmt_double_label(b_val);
    std::string csv_dir = std::string("src/vi_2p/csv/") + params_label + std::string("/ctsvi_ad/");
    std::string cmd = "mkdir -p " + csv_dir; int unused = system(cmd.c_str()); (void)unused;

    write_csv(csv_dir + "q_history.csv", q_history);
    write_csv(csv_dir + "momentum_history.csv", momentum_history);
    write_csv_scalar_series(csv_dir + "time_history.csv", time_history);
    write_csv_scalar_series(csv_dir + "energy_history.csv", energy_history);
    write_csv_scalar_series(csv_dir + "delta_energy_history.csv", delta_energy_history);
    write_csv_scalar_series(csv_dir + "energy_T_history.csv", energy_T_history);
    write_csv_scalar_series(csv_dir + "energy_U_history.csv", energy_U_history);
    write_csv_3d(csv_dir + "ee_history.csv", ee_history);

    std::ofstream avg_time_file(csv_dir + "avg_runtime.txt");
    avg_time_file << avg_time * 1000 << std::endl;
    avg_time_file.close();

    RCLCPP_INFO(node->get_logger(), "Saved CSVs.");

    // rclcpp::shutdown();
    return 0;
}
