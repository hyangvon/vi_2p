// variational_integrator_adaptive.cpp
// Full AD version of ATSVI (Adaptive-step variational integrator) using Pinocchio + CppAD + Eigen + rclcpp
// Created by assistant (merged & fixed). Save as variational_integrator_adaptive.cpp

#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>

// Fix Eigen <-> CppAD compatibility for isfinite()
namespace Eigen {
    namespace numext {
        template<>
        EIGEN_STRONG_INLINE bool isfinite(const CppAD::AD<double>&) {
            return true;
        }
    }
}

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/energy.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <Eigen/Dense>

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

using namespace pinocchio;
using namespace std::chrono;

// ---------- type aliases ----------
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

// ---------- double helpers (non-AD) ----------
Mat inertia_matrix(const Model &model, Data &data, const Vec &q)
{
    pinocchio::crba(model, data, q);
    return data.M;
}

double kinetic_energy(const Model &model, Data &data, const Vec &q_mid, const Vec &qdot)
{
    pinocchio::crba(model, data, q_mid);
    Mat M = data.M;
    return 0.5 * qdot.transpose() * M * qdot;
}

double potential_energy(const Model &model, Data &data, const Vec &q)
{
    return pinocchio::computePotentialEnergy(model, data, q);
}

// double discrete Lagrangian (used for Ed numeric)
double discrete_lagrangian_double(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h)
{
    Vec q_mid = 0.5 * (q0 + q1);
    Vec dq = (q1 - q0) / h;
    double T = kinetic_energy(model, data, q_mid, dq);
    double U = potential_energy(model, data, q_mid);
    return h * (T - U);
}

// numeric discrete energy Ed = -dL/dh (centered difference)
double discrete_energy_numeric(const Model &model, Data &data, const Vec &q0, const Vec &q1, double h, double eps_h = 1e-8)
{
    double Lp = discrete_lagrangian_double(model, data, q0, q1, h + eps_h);
    double Lm = discrete_lagrangian_double(model, data, q0, q1, h - eps_h);
    double dLdh = (Lp - Lm) / (2.0 * eps_h);
    return -dLdh;
}

// ---------- AD discrete Lagrangian ----------
ADScalar discreteLagrangian_ad(const ModelAD &model_ad, DataAD &data_ad,
                               const VecAD &q0_ad, const VecAD &q1_ad, double h)
{
    VecAD q_mid = ADScalar(0.5) * (q0_ad + q1_ad);
    VecAD dq = (q1_ad - q0_ad) / ADScalar(h);

    pinocchio::crba(model_ad, data_ad, q_mid); // fills data_ad.M (AD)
    MatAD M = data_ad.M;

    ADScalar T = ADScalar(0.5) * dq.transpose() * (M * dq);
    ADScalar U = pinocchio::computePotentialEnergy(model_ad, data_ad, q_mid);

    return ADScalar(h) * (T - U);
}

// ---------- AD gradients D1, D2 (returns double vector evaluated at q) ----------
Vec D_1_ad(const ModelAD &model_ad, DataAD &data_ad, const Vec &q0, const Vec &q1, double h)
{
    const int n = q0.size();
    // independent variables: q0
    CppAD::vector<ADScalar> x_ad(n);
    for (int i=0;i<n;++i) x_ad[i] = q0[i];
    CppAD::Independent(x_ad);

    VecAD q0_ad = Eigen::Map<VecAD>(x_ad.data(), n);
    VecAD q1_ad = q1.cast<ADScalar>();

    ADScalar Ld = discreteLagrangian_ad(model_ad, data_ad, q0_ad, q1_ad, h);
    CppAD::vector<ADScalar> y(1);
    y[0] = Ld;

    CppAD::ADFun<double> f;
    f.Dependent(x_ad, y);

    CppAD::vector<double> x0(n);
    for (int i=0;i<n;++i) x0[i] = q0[i];

    CppAD::vector<double> jac = f.Jacobian(x0); // y dim 1 -> gradient length n

    Vec grad(n);
    for (int i=0;i<n;++i) grad[i] = jac[i];
    return grad;
}

Vec D_2_ad(const ModelAD &model_ad, DataAD &data_ad, const Vec &q0, const Vec &q1, double h)
{
    const int n = q1.size();
    // independent variables: q1
    CppAD::vector<ADScalar> x_ad(n);
    for (int i=0;i<n;++i) x_ad[i] = q1[i];
    CppAD::Independent(x_ad);

    VecAD q0_ad = q0.cast<ADScalar>();
    VecAD q1_ad = Eigen::Map<VecAD>(x_ad.data(), n);

    ADScalar Ld = discreteLagrangian_ad(model_ad, data_ad, q0_ad, q1_ad, h);
    CppAD::vector<ADScalar> y(1);
    y[0] = Ld;

    CppAD::ADFun<double> f;
    f.Dependent(x_ad, y);

    CppAD::vector<double> x0(n);
    for (int i=0;i<n;++i) x0[i] = q1[i];

    CppAD::vector<double> jac = f.Jacobian(x0);

    Vec grad(n);
    for (int i=0;i<n;++i) grad[i] = jac[i];
    return grad;
}

// ---------- Solver info ----------
struct SolverInfo { bool converged; std::string reason; int iterations; double residual_norm; };

// ---------- VI_init (initial implicit step) ----------
std::pair<Vec, SolverInfo> VI_init_ad(const Model &model, Data &data,
                                      const ModelAD &model_ad, DataAD &data_ad,
                                      const Vec &q0, const Vec &v0, const Vec &tau_k, double h,
                                      double eps=1e-6, int max_iters=50, double tol=1e-8)
{
    int n = q0.size();
    Vec q1 = q0 + h * v0; // initial guess
    SolverInfo info{false, "", 0, 0.0};

    for (int it=0; it<max_iters; ++it)
    {
        // Residual R = M(q0) * v0 + D1(q0,q1) - h * tau_k
        Mat M = inertia_matrix(model, data, q0);
        Vec D1 = D_1_ad(model_ad, data_ad, q0, q1, h);
        Vec R = M * v0 + D1 - h * tau_k;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q1, info}; }

        // numeric Jacobian of D1 wrt q1 (columns)
        Mat J = Mat::Zero(n, n);
        double eps_j = eps;
        for (int j=0;j<n;++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps_j;
            Vec D1p = D_1_ad(model_ad, data_ad, q0, q1 + dq, h);
            Vec D1m = D_1_ad(model_ad, data_ad, q0, q1 - dq, h);
            J.col(j) = (D1p - D1m) / (2.0 * eps_j);
        }

        Mat A = J + 1e-9 * Mat::Identity(n,n);
        Eigen::ColPivHouseholderQR<Mat> solver(A);
        if (solver.rank() < n)
        {
            info.converged = false;
            info.reason = "singular_jacobian";
            return {q1, info};
        }
        Vec delta = solver.solve(-R);
        q1 += delta;
    }
    info.converged = false;
    info.reason = "max_iters";
    return {q1, info};
}

// ---------- Fixed-step DEL solver (Newton on q_next) ----------
std::pair<Vec, SolverInfo> solve_q_next_fixed_ad(const ModelAD &model_ad, DataAD &data_ad,
                                                 const Vec &q_prev, const Vec &q_curr,
                                                 const Vec &tau_k, double h,
                                                 double eps=1e-6, int max_iters=50, double tol=1e-8)
{
    int n = q_curr.size();
    Vec q_next = q_curr;
    SolverInfo info{false, "", 0, 0.0};

    for (int it=0; it<max_iters; ++it)
    {
        Vec D2 = D_2_ad(model_ad, data_ad, q_prev, q_curr, h);
        Vec D1 = D_1_ad(model_ad, data_ad, q_curr, q_next, h);
        Vec R = D2 + D1 - h * tau_k;
        double normR = R.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q_next, info}; }

        Mat J = Mat::Zero(n, n);
        double eps_j = eps;
        for (int j=0;j<n;++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps_j;
            Vec D1p = D_1_ad(model_ad, data_ad, q_curr, q_next + dq, h);
            Vec D1m = D_1_ad(model_ad, data_ad, q_curr, q_next - dq, h);
            J.col(j) = (D1p - D1m) / (2.0 * eps_j);
        }

        Mat A = J + 1e-9 * Mat::Identity(n,n);
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

// ---------- SEM adaptive-step solver (Newton on [q_next, h_next]) ----------
std::tuple<Vec, double, SolverInfo> solve_q_next_sem_ad(
    const Model &model, Data &data,
    const ModelAD &model_ad, DataAD &data_ad,
    const Vec &q_prev, const Vec &q_curr,
    double h_prev,
    const Vec &qdot_guess,
    const Vec &tau_k,
    double eps_q = 1e-6,
    double eps_h = 1e-6,
    int max_iters = 80,
    double tol = 1e-8,
    double h_min = 1e-6,
    double h_max = 0.1)
{
    int n = q_curr.size();
    Vec q_next = q_curr + h_prev * qdot_guess;
    double h_next = h_prev;

    // E_prev computed via double model (numeric)
    double E_prev = discrete_energy_numeric(model, data, q_prev, q_curr, h_prev, eps_h);

    SolverInfo info{false, "", 0, 0.0};
    Vec Rvec = Vec::Zero(n+1);

    for (int it=0; it<max_iters; ++it)
    {
        // D2 depends on q_prev, q_curr, h_prev
        Vec D2 = D_2_ad(model_ad, data_ad, q_prev, q_curr, h_prev);
        // D1 depends on q_curr, q_next, h_next
        Vec D1 = D_1_ad(model_ad, data_ad, q_curr, q_next, h_next);

        Vec gvec = h_prev * D2 + h_next * D1; // size n
        double E_next = discrete_energy_numeric(model, data, q_curr, q_next, h_next, eps_h);
        double fval = E_prev - E_next;

        for (int i=0;i<n;++i) Rvec(i) = gvec(i);
        Rvec(n) = fval;

        double normR = Rvec.norm();
        info.iterations = it;
        info.residual_norm = normR;
        if (normR < tol) { info.converged = true; return {q_next, h_next, info}; }

        // Build Jacobian J (n+1 x n+1)
        Mat J = Mat::Zero(n+1, n+1);

        // columns 0..n-1: derivatives wrt q_next_j (numeric diffs of D1 and Ed)
        for (int j=0;j<n;++j)
        {
            Vec dq = Vec::Zero(n);
            dq(j) = eps_q;
            Vec D1p = D_1_ad(model_ad, data_ad, q_curr, q_next + dq, h_next);
            Vec D1m = D_1_ad(model_ad, data_ad, q_curr, q_next - dq, h_next);
            Vec col = h_next * (D1p - D1m) / (2.0 * eps_q);
            for (int i=0;i<n;++i) J(i, j) = col(i);

            double Ep = discrete_energy_numeric(model, data, q_curr, q_next + dq, h_next, eps_h);
            double Em = discrete_energy_numeric(model, data, q_curr, q_next - dq, h_next, eps_h);
            double df_dqj = - (Ep - Em) / (2.0 * eps_q);
            J(n, j) = df_dqj;
        }

        // column n: derivative wrt h_next (numeric)
        double dh = eps_h;
        Vec D1p_h = D_1_ad(model_ad, data_ad, q_curr, q_next, h_next + dh);
        Vec D1m_h = D_1_ad(model_ad, data_ad, q_curr, q_next, h_next - dh);
        Vec dD1_dh = (D1p_h - D1m_h) / (2.0 * dh);
        Vec col_h = D1 + h_next * dD1_dh; // ∂g/∂h_next
        for (int i=0;i<n;++i) J(i, n) = col_h(i);

        double Ep_h = discrete_energy_numeric(model, data, q_curr, q_next, h_next + dh, eps_h);
        double Em_h = discrete_energy_numeric(model, data, q_curr, q_next, h_next - dh, eps_h);
        double dEd_dh = (Ep_h - Em_h) / (2.0 * dh);
        J(n, n) = - dEd_dh;

        // Solve linear system
        Mat Jreg = J + 1e-9 * Mat::Identity(n+1, n+1);
        Eigen::ColPivHouseholderQR<Mat> solver(Jreg);
        if (solver.rank() < n+1)
        {
            info.converged = false;
            info.reason = "singular_jacobian";
            return {q_next, h_next, info};
        }

        Vec delta = solver.solve(-Rvec);
        Vec dq = delta.head(n);
        double dh_scalar = delta(n);

        // Backtracking line search
        double alpha = 1.0;
        bool accept = false;
        for (int ls=0; ls<10; ++ls)
        {
            Vec q_trial = q_next + alpha * dq;
            double h_trial = h_next + alpha * dh_scalar;
            if (h_trial <= h_min || h_trial >= h_max) { alpha *= 0.5; continue; }

            Vec D1_trial = D_1_ad(model_ad, data_ad, q_curr, q_trial, h_trial);
            Vec g_trial = h_prev * D2 + h_trial * D1_trial;
            double E_trial = discrete_energy_numeric(model, data, q_curr, q_trial, h_trial, eps_h);
            double f_trial = E_prev - E_trial;

            Vec Rtrial(n+1);
            for (int i=0;i<n;++i) Rtrial(i) = g_trial(i);
            Rtrial(n) = f_trial;

            if (Rtrial.norm() < (1.0 - 1e-4 * alpha) * normR || alpha < 1e-3)
            {
                q_next = q_trial;
                h_next = h_trial;
                accept = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!accept)
        {
            info.converged = false;
            info.reason = "line_search_failed";
            info.iterations = it;
            info.residual_norm = normR;
            return {q_next, h_next, info};
        }
    }

    info.converged = false;
    info.reason = "max_iters";
    return {q_next, h_next, info};
}

// Optional: ABA dynamics step helper
std::pair<Vec, Vec> pinocchio_dynamics_step(const Model &model, Data &data, const Vec &q, const Vec &v, const Vec &tau, double h)
{
    pinocchio::aba(model, data, q, v, tau);
    Vec qdd = data.ddq;
    Vec v_next = v + h * qdd;
    Vec q_next = q + h * v_next;
    return {q_next, v_next};
}

// CSV writers
void write_csv(const std::string &filename, const std::vector<Vec> &rows)
{
    if (rows.empty()) return;
    std::ofstream ofs(filename);
    int ncols = rows[0].size();
    for (size_t r=0;r<rows.size();++r)
    {
        for (int c=0;c<ncols;++c)
        {
            ofs << std::setprecision(15) << rows[r](c);
            if (c+1 < ncols) ofs << ",";
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
    std::cout.setf(std::ios::unitbuf);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>(
        "atsvi_ad_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // Parameters (provide defaults via launch)
    std::vector<double> q_init_vec;
    double q_init_scalar = 0.2;
    bool q_init_is_vec = node->get_parameter("q_init", q_init_vec);
    if (!q_init_is_vec) node->get_parameter("q_init", q_init_scalar);
    double timestep = node->get_parameter("timestep").as_double();
    double duration = node->get_parameter("duration").as_double();
    double eps_diff = node->get_parameter("eps_diff").as_double();
    // std::string urdf_path = node->get_parameter("urdf_path").as_string();
    std::string urdf_path = expand_user(node->get_parameter("urdf_path").as_string());
    double h_min = node->get_parameter("h_min").as_double();
    double h_max = node->get_parameter("h_max").as_double();
    int max_adapt_iters = node->get_parameter("max_adapt_iters").as_int();

    RCLCPP_INFO(node->get_logger(), "Using URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(node->get_logger(), "Duration = %.3f, Timestep(init) = %.6f, eps_diff = %.1e", duration, timestep, eps_diff);
    RCLCPP_INFO(node->get_logger(), "h_min=%.1e h_max=%.6f max_adapt_iters=%d", h_min, h_max, max_adapt_iters);

    // load model (double) and data
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

    RCLCPP_INFO(node->get_logger(), "Model has nq=%d nv=%d", model.nq, model.nv);

    // build AD model/data
    ModelAD model_ad = model.cast<ADScalar>();
    DataAD data_ad(model_ad);

    int n = model.nq;
    int n_steps = static_cast<int>(duration / timestep);

    // initial conditions
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
    Vec tau_k = Vec::Zero(n); // currently zero torques; modify as needed

    std::vector<Vec> q_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> delta_energy_history;
    std::vector<double> h_history;
    std::vector<double> runtimes;
    std::vector<Eigen::Vector3d> ee_history;
    std::vector<Vec> momentum_history;

    time_history.push_back(0.0);
    h_history.push_back(timestep);
    q_history.push_back(q_prev);

    auto t_start = high_resolution_clock::now();

    // initial energy
    Mat M0 = inertia_matrix(model, data, q_prev);
    Vec qdot0 = v_prev;
    double T0 = 0.5 * qdot0.transpose() * M0 * qdot0;
    double U0 = potential_energy(model, data, q_prev);
    double total_energy = T0 + U0;
    energy_history.push_back(total_energy);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    auto [ee_pos0, ee_rot0] = compute_end_effector_pose(model, data, link_tcp_id, q_prev);
    ee_history.push_back(ee_pos0);

    // initial implicit VI step (use VI_init_ad)
    // auto [q_curr, info_init] = VI_init_ad(model, data, model_ad, data_ad, q_prev, v_prev, tau_k, timestep, eps_diff);
    // if (!info_init.converged) {
    //     RCLCPP_WARN(node->get_logger(), "VI_init did not converge (reason=%s). Proceeding with initial guess.", info_init.reason.c_str());
    // }
    // q_history.push_back(q_curr);
    //
    // double T = kinetic_energy(model, data, (q_curr + q_prev) / 2.0, (q_curr - q_prev)/timestep);
    // double U = potential_energy(model, data, (q_curr + q_prev) / 2.0);
    // total_energy = T + U;
    // energy_history.push_back(total_energy);
    // delta_energy_history.push_back(energy_history.back() - energy_history.front());
    
    auto [q_curr, h_next, info_adapt] = solve_q_next_sem_ad(
            model, data, model_ad, data_ad,
            q_prev,
            q_prev + v_prev * timestep,
            timestep,
            v_prev,
            tau_k,
            /*eps_q=*/eps_diff,
            /*eps_h=*/eps_diff,
            /*max_iters=*/max_adapt_iters,
            /*tol=*/1e-8,
            h_min,
            h_max
        );
        
    q_history.push_back(q_curr);
    
    double T = kinetic_energy(model, data, (q_curr + q_prev) / 2.0, (q_curr - q_prev)/timestep);
    double U = potential_energy(model, data, (q_curr + q_prev) / 2.0);
    total_energy = T + U;
    energy_history.push_back(total_energy);
    delta_energy_history.push_back(energy_history.back() - energy_history.front());

    auto [ee_pos1, _ee_rot1] = compute_end_effector_pose(model, data, link_tcp_id, q_curr);
    ee_history.push_back(ee_pos1);

    double t_cur = 0.0;
    t_cur += timestep; // we have advanced to q_curr at t = timestep
    time_history.push_back(timestep);
    h_history.push_back(timestep);

    // int max_steps = std::max((int)(duration / h_min) + 10, 1000);
    for (int step=0; step < n_steps -1  && t_cur < duration - 1e-12; ++step)
    {
        auto t0 = high_resolution_clock::now();

        Vec qdot_guess = (q_history.back() - q_history[q_history.size()-2]) / h_history.back();

        auto [q_next, h_next, info_adapt] = solve_q_next_sem_ad(
            model, data, model_ad, data_ad,
            q_history[q_history.size()-2],
            q_history[q_history.size()-1],
            h_history.back(),
            qdot_guess,
            tau_k,
            /*eps_q=*/eps_diff,
            /*eps_h=*/eps_diff,
            /*max_iters=*/max_adapt_iters,
            /*tol=*/1e-8,
            h_min,
            h_max
        );

        if (!info_adapt.converged)
        {
            RCLCPP_WARN(node->get_logger(), "Step %d: SEM solver failed (%s). Falling back to fixed-step DEL.", step, info_adapt.reason.c_str());
            auto [q_fix, info_fix] = solve_q_next_fixed_ad(model_ad, data_ad, q_history[q_history.size()-2], q_history[q_history.size()-1], tau_k, h_history.back(), eps_diff);
            q_next = q_fix;
            h_next = h_history.back();
            RCLCPP_INFO(node->get_logger(), "Fixed-step info: converged=%d it=%d res=%f reason=%s", info_fix.converged, info_fix.iterations, info_fix.residual_norm, info_fix.reason.c_str());
        }

        q_history.push_back(q_next);
        h_history.push_back(h_next);
        t_cur += h_next;
        time_history.push_back(t_cur);

        T = kinetic_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0, (q_history.back() - q_history[q_history.size()-2]) / h_next);
        U = potential_energy(model, data, (q_history.back() + q_history[q_history.size()-2]) / 2.0);
        total_energy = T + U;
        energy_history.push_back(total_energy);
        delta_energy_history.push_back(energy_history.back() - energy_history.front());

        auto [ee_pos, ee_rot] = compute_end_effector_pose(model, data, link_tcp_id, q_next);
        ee_history.push_back(ee_pos);

        // compute generalized momentum p = M(q_mid) * qdot
        Vec qdot = (q_next - q_history[q_history.size()-2]) / h_next;
        Vec qmid = 0.5 * (q_next + q_history[q_history.size()-2]);
        Mat Mmid = inertia_matrix(model, data, qmid);
        Vec p = Mmid * qdot;
        momentum_history.push_back(p);

        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        double progress = t_cur / duration;
        if (progress > 1.0) progress = 1.0;
        int bar_width = 50;
        int pos = int(bar_width * progress);
        double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
        double time_left = avg_time * (duration - t_cur) / std::max(h_min, h_next);
        std::cout << "\r[";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%, approx " << int(time_left/60) << "mins " << int(time_left) % 60 << "s left... ";
        std::cout.flush();
    }
    std::cout << "\n";

    // if (!time_history.empty()) {
    //     time_history.pop_back();
    // }
    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
    RCLCPP_INFO(node->get_logger(), "Simulation finished, wall time: %f s, Average step time: %f ms", total_elapsed, avg_time*1e3);

    // Save CSVs into parameterized folder including lyap params: src/vi_2p/csv/q<q>_dt<dt>_T<T>_a<alpha>_b<beta>/atsvi_ad/
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
    std::string csv_dir = std::string("src/vi_2p/csv/") + params_label + std::string("/atsvi_ad/");
    std::string cmd = "mkdir -p " + csv_dir; int unused = system(cmd.c_str()); (void)unused;

    write_csv(csv_dir + "q_history.csv", q_history);
    write_csv(csv_dir + "momentum_history.csv", momentum_history);
    write_csv_scalar_series(csv_dir + "time_history.csv", time_history);
    write_csv_scalar_series(csv_dir + "energy_history.csv", energy_history);
    write_csv_scalar_series(csv_dir + "h_history.csv", h_history);
    write_csv_scalar_series(csv_dir + "delta_energy_history.csv", delta_energy_history);
    write_csv_3d(csv_dir + "ee_history.csv", ee_history);

    std::ofstream avg_time_file(csv_dir + "avg_runtime.txt");
    avg_time_file << avg_time * 1000 << std::endl;
    avg_time_file.close();

    RCLCPP_INFO(node->get_logger(), "Saved CSVs.");

    return 0;
}
