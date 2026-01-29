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
using Vec  = Eigen::VectorXd;
using Mat  = Eigen::MatrixXd;

// Pinocchio timing diagnostics
static std::vector<double> pin_runtimes;
static int pin_error_count = 0;

// 计算末端位姿（位置 + 旋转矩阵）
std::pair<Eigen::Vector3d, Eigen::Matrix3d>
compute_end_effector_pose(const Model &model,
                          Data &data,
                          int frame_id,
                          const Vec &q)
{
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    Eigen::Vector3d ee_pos = data.oMf[frame_id].translation();
    Eigen::Matrix3d ee_rot = data.oMf[frame_id].rotation();

    return {ee_pos, ee_rot};
}

// ---------- double helpers (non-AD) ----------
Mat inertia_matrix(const Model &model, Data &data, const Vec &q)
{
    try {
        auto t0 = high_resolution_clock::now();
        pinocchio::crba(model, data, q);
        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        pin_runtimes.push_back(elapsed);
    } catch (const std::exception &e) {
        pin_error_count++;
    }
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

// 计算总能量 (动能 + 势能)
double compute_total_energy(const Model &model, Data &data, const Vec &q, const Vec &qdot)
{
    double T = kinetic_energy(model, data, q, qdot);
    double U = potential_energy(model, data, q);
    return T + U;
}

// ---------- RK4 integrator state ----------
struct State {
    Vec q;  // 广义坐标
    Vec v;  // 广义速度
};

// 动力学模型：计算加速度
// 使用 ABA (Articulated Body Algorithm) 直接计算：a = M^{-1} * (tau - C*v - g)
State dynamics(const Model &model, Data &data, const State &state, const Vec &tau)
{
    const Vec &q = state.q;
    const Vec &v = state.v;

    // 使用 ABA 算法计算加速度
    try {
        auto t0 = high_resolution_clock::now();
        pinocchio::aba(model, data, q, v, tau);
        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        pin_runtimes.push_back(elapsed);
    } catch (const std::exception &e) {
        pin_error_count++;
    }
    Vec a = data.ddq;

    State ds;
    ds.q = v;
    ds.v = a;
    return ds;
}

// 向量加法辅助函数
State add_state(const State &s1, const State &s2)
{
    State result;
    result.q = s1.q + s2.q;
    result.v = s1.v + s2.v;
    return result;
}

// 向量标量乘法辅助函数
State scale_state(const State &s, double c)
{
    State result;
    result.q = c * s.q;
    result.v = c * s.v;
    return result;
}

// RK4 单步积分
State rk4_step(const Model &model, Data &data, const State &y_n, const Vec &tau, double h)
{
    // k1 = f(t_n, y_n)
    State k1 = dynamics(model, data, y_n, tau);

    // k2 = f(t_n + h/2, y_n + h/2 * k1)
    State y_mid1 = add_state(y_n, scale_state(k1, h / 2.0));
    State k2 = dynamics(model, data, y_mid1, tau);

    // k3 = f(t_n + h/2, y_n + h/2 * k2)
    State y_mid2 = add_state(y_n, scale_state(k2, h / 2.0));
    State k3 = dynamics(model, data, y_mid2, tau);

    // k4 = f(t_n + h, y_n + h * k3)
    State y_end = add_state(y_n, scale_state(k3, h));
    State k4 = dynamics(model, data, y_end, tau);

    // y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    State y_next = y_n;
    y_next.q = y_n.q + (h / 6.0) * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q);
    y_next.v = y_n.v + (h / 6.0) * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);

    return y_next;
}

// CSV writers
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
    for (double v : rows) 
        ofs << std::setprecision(15) << v << "\n";
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

void print_progress(double t_cur, double duration, std::vector<double> runtimes, double h_step) {
    int bar_width = 50;
    double progress = t_cur / duration;
    if (progress > 1.0) progress = 1.0;
    int pos = int(bar_width * progress);
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
    double time_left = avg_time * (duration - t_cur) / h_step;
    std::cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%, approx " << int(time_left / 60) << "mins " << int(time_left) % 60 << "s left... ";
    std::cout.flush();
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
    // trim trailing zeros and possible trailing dot
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
        "rk4_node",
        rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
    );

    // Parameters (provide defaults via launch)
    std::vector<double> q_init_vec;
    double q_init_scalar = 0.2;
    bool q_init_is_vec = node->get_parameter("q_init", q_init_vec);
    if (!q_init_is_vec) {
        // try scalar
        node->get_parameter("q_init", q_init_scalar);
    }
    double timestep = node->get_parameter("timestep").as_double();
    double duration = node->get_parameter("duration").as_double();
    double eps_diff = node->get_parameter("eps_diff").as_double();
    std::string urdf_path = expand_user(node->get_parameter("urdf_path").as_string());

    RCLCPP_INFO(node->get_logger(), "Using URDF: %s", urdf_path.c_str());
    RCLCPP_INFO(node->get_logger(), "Duration = %.3f, Timestep = %.6f, eps_diff = %.1e", duration, timestep, eps_diff);

    // load model (double) and data
    Model model;
    try {
        pinocchio::urdf::buildModel(urdf_path, model);
    } catch (const std::exception &e) {
        RCLCPP_ERROR(node->get_logger(), "Error loading URDF: %s", e.what());
        return 1;
    }
    Data data(model);
    model.gravity.linear(Eigen::Vector3d(0, 0, -9.81));

    // 获取末端 frame id
    int link_tcp_id = model.getFrameId("link_tcp");
    if (link_tcp_id == -1) {
        RCLCPP_ERROR(node->get_logger(), "TCP frame not found!");
        return 1;
    }
    RCLCPP_INFO(node->get_logger(), "Model has nq=%d, nv=%d", model.nq, model.nv);

    int n = model.nq;
    int n_steps = static_cast<int>(duration / timestep);

    // initial conditions
    State state;
    if (q_init_is_vec) {
        state.q = Vec::Zero(n);
        for (int i = 0; i < n; ++i) {
            if (i < static_cast<int>(q_init_vec.size())) state.q(i) = q_init_vec[i];
            else state.q(i) = q_init_vec.back();
        }
    } else {
        state.q = Vec::Constant(n, q_init_scalar);
    }
    state.v = Vec::Zero(n);
    Vec tau = Vec::Zero(n);  // 控制扭矩 (目前设为零)

    // 历史记录
    std::vector<Vec> q_history;
    std::vector<Vec> v_history;
    std::vector<Vec> momentum_history;
    std::vector<double> time_history;
    std::vector<double> energy_history;
    std::vector<double> delta_energy_history;
    std::vector<double> runtimes;
    std::vector<Eigen::Vector3d> ee_history;

    q_history.push_back(state.q);
    v_history.push_back(state.v);
    time_history.push_back(0.0);

    auto t_start = high_resolution_clock::now();

    // 初始能量
    double E_init = compute_total_energy(model, data, state.q, state.v);
    energy_history.push_back(E_init);
    delta_energy_history.push_back(0.0);

    auto [ee_pos0, ee_rot0] = compute_end_effector_pose(model, data, link_tcp_id, state.q);
    ee_history.push_back(ee_pos0);

    // RK4 仿真循环
    double t_cur = 0.0;
    for (int step = 0; step < n_steps && t_cur < duration - 1e-12; ++step)
    {
        auto t0 = high_resolution_clock::now();

        // 单步 RK4 积分
        State state_next = rk4_step(model, data, state, tau, timestep);
        state = state_next;

        t_cur += timestep;
        time_history.push_back(t_cur);
        q_history.push_back(state.q);
        v_history.push_back(state.v);

        // compute generalized momentum p = M(q_mid) * qdot
        Vec qmid = state.q;
        if (q_history.size() > 1) qmid = 0.5 * (state.q + q_history[q_history.size()-2]);
        Vec qdot = state.v;
        Mat Mmid = inertia_matrix(model, data, qmid);
        Vec p = Mmid * qdot;
        momentum_history.push_back(p);

        // 能量计算
        double E_cur = compute_total_energy(model, data, state.q, state.v);
        energy_history.push_back(E_cur);
        delta_energy_history.push_back(E_cur - E_init);

        // 末端位置
        auto [ee_pos, ee_rot] = compute_end_effector_pose(model, data, link_tcp_id, state.q);
        ee_history.push_back(ee_pos);

        auto t1 = high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        runtimes.push_back(elapsed);

        print_progress(t_cur, duration, runtimes, timestep);
    }
    std::cout << "\n";

    auto t_end = high_resolution_clock::now();
    double total_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
    double avg_time = !runtimes.empty() ? std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size() : 0.0;
    RCLCPP_INFO(node->get_logger(), "Simulation finished, wall time: %f s, Average step time: %f ms", total_elapsed, avg_time * 1e3);

    // Save CSVs into parameterized folder including lyap params: src/vi_2p/csv/q<q>_dt<dt>_T<T>_a<alpha>_b<beta>/rk4/
    auto [a_val, b_val] = read_lyap_from_config();
    std::string q_label;
    if (q_init_is_vec) {
        std::ostringstream qqs;
        for (size_t i = 0; i < q_init_vec.size(); ++i) {
            if (i) qqs << "_";
            qqs << fmt_double_label(q_init_vec[i]);
        }
        q_label = qqs.str();
    } else {
        q_label = fmt_double_label(q_init_scalar);
    }

    std::string params_label = std::string("q") + q_label
        + std::string("_dt") + fmt_double_label(timestep)
        + std::string("_T") + fmt_double_label(duration)
        + std::string("_a") + fmt_double_label(a_val)
        + std::string("_b") + fmt_double_label(b_val);
    std::string csv_dir = std::string("src/vi_2p/csv/") + params_label + std::string("/rk4/");
    std::string cmd = "mkdir -p " + csv_dir;
    int unused = system(cmd.c_str()); (void)unused;

    write_csv(csv_dir + "q_history.csv", q_history);
    write_csv(csv_dir + "v_history.csv", v_history);
    write_csv(csv_dir + "momentum_history.csv", momentum_history);
    write_csv_scalar_series(csv_dir + "time_history.csv", time_history);
    write_csv_scalar_series(csv_dir + "energy_history.csv", energy_history);
    write_csv_scalar_series(csv_dir + "delta_energy_history.csv", delta_energy_history);
    write_csv_3d(csv_dir + "ee_history.csv", ee_history);

    std::ofstream avg_time_file(csv_dir + "avg_runtime.txt");
    avg_time_file << avg_time * 1000 << std::endl;  // 保存为毫秒单位
    avg_time_file.close();

    // Save Pinocchio-only average runtime if we collected any timings
    try {
        if (!pin_runtimes.empty()) {
            double avg_pin = std::accumulate(pin_runtimes.begin(), pin_runtimes.end(), 0.0) / pin_runtimes.size();
            std::ofstream pin_time_file(csv_dir + "avg_runtime_pin.txt");
            pin_time_file << (avg_pin * 1000.0) << std::endl; // ms
            pin_time_file.close();
            RCLCPP_INFO(node->get_logger(), "Saved Pinocchio avg per-step time: %f ms (samples=%zu, errors=%d)", avg_pin*1000.0, pin_runtimes.size(), pin_error_count);
        } else {
            RCLCPP_INFO(node->get_logger(), "No Pinocchio timing samples collected.");
        }
    } catch (const std::exception &e) {
        RCLCPP_WARN(node->get_logger(), "Failed to write Pinocchio timing file: %s", e.what());
    }

    RCLCPP_INFO(node->get_logger(), "Data saved to %s", csv_dir.c_str());

    return 0;
}
