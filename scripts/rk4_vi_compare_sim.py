#!/usr/bin/env python3
import subprocess
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import yaml
except Exception:
    yaml = None

# 绘图统一风格与辅助函数
FIGSIZE = (10, 5)
DEFAULT_DPI = 200

# 字体与样式统一设置
FONT_FAMILY = 'DejaVu Sans'
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
TICK_FONT_SIZE = 18
TITLE_FONT_WEIGHT = 'bold'

def _init_fig(figsize=None):
    if figsize is None:
        figsize = FIGSIZE
    # 通过 rcParams 统一字体、标题和图例样式
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.titleweight': TITLE_FONT_WEIGHT,
        'axes.labelsize': LABEL_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'xtick.labelsize': TICK_FONT_SIZE,
        'ytick.labelsize': TICK_FONT_SIZE,
        'legend.frameon': True,
    })
    plt.figure(figsize=figsize)

def _save_fig(tag, filename, dpi, show=True):
    params_label = build_params_label()
    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi_2p/fig/{params_label}/{tag}")
    os.makedirs(save_dir, exist_ok=True)

    # 读取配置并将关键参数加入文件名
    config_path = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')
    q_init = 'NA'
    timestep = 'NA'
    duration = 'NA'
    lyap_alpha = 'NA'
    lyap_beta = 'NA'
    try:
        if yaml is not None:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            root = cfg.get('/**', {}) if isinstance(cfg, dict) else {}
            rp = root.get('ros__parameters', {}) if isinstance(root, dict) else {}
            q_init = rp.get('q_init', q_init)
            timestep = rp.get('timestep', timestep)
            duration = rp.get('duration', duration)
            ets = cfg.get('etsvi_node', {})
            ets_rp = ets.get('ros__parameters', {}) if isinstance(ets, dict) else {}
            lyap_alpha = ets_rp.get('lyap_alpha', lyap_alpha)
            lyap_beta = ets_rp.get('lyap_beta', lyap_beta)
        else:
            with open(config_path, 'r') as f:
                txt = f.read()
            import re
            def _find(k):
                m = re.search(rf"{k}\s*:\s*([0-9.eE+-]+)", txt)
                return m.group(1) if m else 'NA'
            q_init = _find('q_init')
            timestep = _find('timestep')
            duration = _find('duration')
            lyap_alpha = _find('lyap_alpha')
            lyap_beta = _find('lyap_beta')
    except Exception:
        pass

    def _format_param(v):
        def _fmt_single(x):
            try:
                fx = float(x)
                s = f"{fx:.2f}"
                # trim trailing zeros and optional dot, then replace '.' with 'p'
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
                    if '.' in s:
                        s = s.replace('.', 'p')
                return s.replace(' ', '')
            except Exception:
                s = str(x)
                s = s.replace('.', 'p').replace(' ', '')
                return s

        if isinstance(v, (list, tuple)):
            parts = [_fmt_single(x) for x in v]
            return "_".join(parts)
        return _fmt_single(v)

    params_str = f"q{_format_param(q_init)}_dt{_format_param(timestep)}_T{_format_param(duration)}_a{_format_param(lyap_alpha)}_b{_format_param(lyap_beta)}"
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_{params_str}{ext}"
    save_path = os.path.join(save_dir, new_filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def build_params_label(include_lyap=True):
    config_path = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')
    q_init = 'NA'
    timestep = 'NA'
    duration = 'NA'
    lyap_alpha = None
    lyap_beta = None
    try:
        if yaml is not None:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            root = cfg.get('/**', {}) if isinstance(cfg, dict) else {}
            rp = root.get('ros__parameters', {}) if isinstance(root, dict) else {}
            q_init = rp.get('q_init', q_init)
            timestep = rp.get('timestep', timestep)
            duration = rp.get('duration', duration)
            ets = cfg.get('etsvi_node', {})
            ets_rp = ets.get('ros__parameters', {}) if isinstance(ets, dict) else {}
            lyap_alpha = ets_rp.get('lyap_alpha', None)
            lyap_beta = ets_rp.get('lyap_beta', None)
        else:
            with open(config_path, 'r') as f:
                txt = f.read()
            import re
            def _find(k):
                m = re.search(rf"{k}\s*:\s*([0-9.eE+-]+)", txt)
                return m.group(1) if m else None
            q_init = _find('q_init') or q_init
            timestep = _find('timestep') or timestep
            duration = _find('duration') or duration
            lyap_alpha = _find('lyap_alpha')
            lyap_beta = _find('lyap_beta')
    except Exception:
        pass

    def _format_param(v):
        def _fmt_single(x):
            try:
                fx = float(x)
                s = f"{fx:.2f}"
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
                    if '.' in s:
                        s = s.replace('.', 'p')
                return s.replace(' ', '')
            except Exception:
                s = str(x)
                s = s.replace('.', 'p').replace(' ', '')
                return s

        if isinstance(v, (list, tuple)):
            parts = [_fmt_single(x) for x in v]
            return "_".join(parts)
        return _fmt_single(v)

    label = f"q{_format_param(q_init)}_dt{_format_param(timestep)}_T{_format_param(duration)}"
    if include_lyap:
        a = lyap_alpha if lyap_alpha is not None else 'NA'
        b = lyap_beta if lyap_beta is not None else 'NA'
        label += f"_a{_format_param(a)}_b{_format_param(b)}"
    return label


def run_ctsvi():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')

    print("Starting ctsvi simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi_2p', 'ctsvi_ad_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def run_atsvi():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')

    print("Starting atsvi simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi_2p', 'atsvi_ad_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def run_etsvi():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')

    print("Starting etsvi simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi_2p', 'etsvi_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def run_rk4():
    """运行 RK4 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')

    print("Starting rk4 simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi_2p', 'rk4_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def run_pybullet_inline():
    """在当前 Python 进程中加载并运行 scripts/pybullet_sim.py 的 main()，便于一键生成 CSV 数据。"""
    script_path = os.path.join(os.path.dirname(__file__), 'pybullet_sim.py')
    if not os.path.exists(script_path):
        print(f"pybullet script not found: {script_path}")
        return False

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('pybullet_sim', script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            print('Running pybullet_sim.main()...')
            module.main()
            return True
        else:
            print('pybullet_sim.py does not define main()')
            return False
    except Exception as e:
        import traceback
        print(f'Error running pybullet inline: {e}')
        print(traceback.format_exc())
        return False

def plot_runtime_comparison(tag, dpi_set):
    """绘制ETSVI和RK4的平均运行时间对比图"""
    print("Generating runtime comparison plot...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params = build_params_label()

    # 定义算法和对应的文件路径（位于参数化目录下）
    algorithms = {
        'etsvi': os.path.join(params, 'etsvi', 'avg_runtime.txt'),
        'rk4': os.path.join(params, 'rk4', 'avg_runtime.txt'),
        'pybullet': os.path.join(params, 'pybullet', 'avg_runtime.txt')
    }

    avg_times = []
    labels = []

    for alg_name, alg_file in algorithms.items():
        try:
            with open(os.path.join(base, alg_file), 'r') as f:
                avg_time = float(f.read().strip())
                avg_times.append(avg_time)
                labels.append(alg_name.upper())
        except FileNotFoundError:
            print(f"Warning: {alg_file} not found, using default value 0")
            avg_times.append(0)
            labels.append(alg_name.upper())

    # 绘制柱状图（使用统一风格）
    _init_fig(figsize=(10, 6))
    bars = plt.bar(labels, avg_times,
                   color=['lightcoral', 'lightyellow', 'lightblue'][:len(labels)],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)

    # 在柱子上添加数值标签
    max_val = max(avg_times) if avg_times else 0
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01,
                 f'{time_val:.3f}ms',
                 ha='center',
                 va='bottom',
                 fontsize=12,
                 fontweight='bold')

    plt.xlabel('Algorithms', fontsize=12)
    plt.ylabel('Average Runtime (ms)', fontsize=12)
    plt.title('Average Runtime Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # 保存并显示（统一）
    filename = f"runtime_comparison_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=True)

    # 显示数值
    for label, time_val in zip(labels, avg_times):
        print(f"{label}: {time_val:.3f}ms")

def plot_results(tag, dpi_set):
    """绘制仿真结果（包含 RK4）"""
    print("Generating plots...")

    # ---------- 1. 读取数据 ----------
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label()

    csv_dir_etsvi = os.path.join(base, params_base, 'etsvi')
    csv_dir_rk4   = os.path.join(base, params_base, 'rk4')
    csv_dir_py    = os.path.join(base, params_base, 'pybullet')

    if not os.path.exists(os.path.join(csv_dir_rk4, 'q_history.csv')):
        print("CSV files not found. Simulation may have failed.")
        return False

    # 读取 CSV（支持可选的 pybullet 目录）
    try:
        tcp_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'ee_history.csv'), delimiter=',')
        tcp_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'ee_history.csv'), delimiter=',')

        delta_energy_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'delta_energy_history.csv'), delimiter=',')

        time_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
        time_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'time_history.csv'), delimiter=',')

        # optional pybullet
        tcp_py = None
        delta_energy_py = None
        time_py = None
        if os.path.exists(os.path.join(csv_dir_py, 'ee_history.csv')):
            tcp_py = np.loadtxt(os.path.join(csv_dir_py, 'ee_history.csv'), delimiter=',')
        if os.path.exists(os.path.join(csv_dir_py, 'delta_energy_history.csv')):
            delta_energy_py = np.loadtxt(os.path.join(csv_dir_py, 'delta_energy_history.csv'), delimiter=',')
        if os.path.exists(os.path.join(csv_dir_py, 'time_history.csv')):
            time_py = np.loadtxt(os.path.join(csv_dir_py, 'time_history.csv'), delimiter=',')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return False

    # ---------- 绘制能量曲线 ----------
    _init_fig()
    # 使用调色板和更明显的样式以增强可区分度
    # cmap = plt.get_cmap('tab10')
    # c_rk4 = cmap(0)
    # c_py = cmap(1)
    # c_etsvi = cmap(2)
    
    c_rk4 = '#1f77b4'
    c_py = '#ff7f0e'
    c_etsvi = '#d62728'

    # 计算采样间隔以减少标记密度
    def _markevery(arr, target=50):
        try:
            n = max(1, int(len(arr) / target))
        except Exception:
            n = 1
        return n

    # RK4
    plt.plot(time_rk4, delta_energy_rk4, label='ΔEnergy of RK4', color=c_rk4,
             linestyle='--', linewidth=1.5)

    # PyBullet (if available)
    if 'delta_energy_py' in locals() and delta_energy_py is not None and time_py is not None:
        plt.plot(time_py, delta_energy_py, label='ΔEnergy of PyBullet', color=c_py,
                 linestyle='-.', linewidth=1.5)
    
    # ETSVI
    plt.plot(time_etsvi, delta_energy_etsvi, label='ΔEnergy of C-ATSVI', color=c_etsvi,
             linestyle='-', linewidth=2.0)
             
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy evolution')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-25, 60)
    filename = f"energy_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    # ---------- 绘制时间步 ----------
    
    # ---------- TCP 位置曲线 ----------
    _init_fig()
    # RK4
    # plt.plot(time_rk4, tcp_rk4[:, 0], label='px_rk4', color=c_rk4, linestyle=':', linewidth=1.5,
    #          marker='s', markersize=3, markevery=_markevery(time_rk4))
    plt.plot(time_rk4, tcp_rk4[:, 2], label='position Z of RK4', color=c_rk4, linestyle='--', linewidth=1.5,
             marker=None)

    # PyBullet
    if 'tcp_py' in locals() and tcp_py is not None:
        # tcp_py may be (N,3)
        # plt.plot(time_py, tcp_py[:, 0], label='px_pybullet', color=c_py, linestyle='-.', linewidth=1.5,
        #          marker='o', markersize=3, markevery=_markevery(time_py), alpha=0.9)
        plt.plot(time_py, tcp_py[:, 2], label='position Z of PyBullet', color=c_py, linestyle='-.', linewidth=1.5,
                 marker=None)

    # ETSVI (TCP position)
    # plt.plot(time_etsvi, tcp_etsvi[:, 0], label='px_etsvi', color=c_etsvi, linestyle='--', linewidth=1.5,
    #          marker='^', markersize=3, markevery=_markevery(time_etsvi))
    plt.plot(time_etsvi, tcp_etsvi[:, 2], label='position Z of C-ATSVI', color=c_etsvi, linestyle='-', linewidth=1.5,
             marker=None)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Tip Position')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-2.5, 1)
    filename = f"tcp_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    plot_runtime_comparison(tag, dpi_set)
    # 绘制 ETSVI / RK4 / PyBullet 的第7关节动量对比图
    try:
        plot_momentum_compare_joint(tag, dpi_set, joint_idx=2)
    except Exception as e:
        print(f"Warning: failed to plot momentum comparison: {e}")
    
    # 绘制第7关节相平面（关节编号从1开始，脚本内部使用0-based）
    # plot_phase_plane("etsvi_rk4_compare", 300, joint_index=6)
    # 绘制 RK4 与 PyBullet 的第2关节相图（分别保存）
    try:
        # plot_rk4_pybullet_joint_phase_separate(tag, dpi_set, joint_idx=2)
        plot_rk4_phase(tag, dpi_set, joint_idx=2)
        plot_pybullet_phase(tag, dpi_set, joint_idx=2)
    except Exception as e:
        print(f"Warning: failed to plot separate RK4/PyBullet phase plots: {e}")

    # 绘制第7关节庞加莱截面：用关节1过零上升事件触发采样
    # plot_poincare_section("etsvi_rk4_compare", 300, joint_index=6, trigger_joint_index=0, surface='q=0', direction='+')

    return 1


def plot_momentum_compare_joint(tag, dpi_set, joint_idx=2):
    """
    在 RK4 比较脚本中绘制 ETSVI / RK4 / PyBullet 的关节动量对比（默认关节7）。
    搜索参数化 CSV 目录下的 momentum_history.csv 与 time_history.csv。
    """
    print(f"Generating joint {joint_idx} momentum comparison (ETSVI/RK4/PyBullet)...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params = build_params_label()
    paths = {
        'C-ATSVI': os.path.join(base, params, 'etsvi', 'momentum_history.csv'),
        'RK4': os.path.join(base, params, 'rk4', 'momentum_history.csv'),
        'PyBullet': os.path.join(base, params, 'pybullet', 'momentum_history.csv')
    }
    times = {
        'C-ATSVI': os.path.join(base, params, 'etsvi', 'time_history.csv'),
        'RK4': os.path.join(base, params, 'rk4', 'time_history.csv'),
        'PyBullet': os.path.join(base, params, 'pybullet', 'time_history.csv')
    }

    _init_fig(figsize=(7,4))
    cmap = plt.get_cmap('tab10')
    colmap = {'C-ATSVI': cmap(2), 'RK4': cmap(0), 'PyBullet': cmap(1)}
    any_plotted = False
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"Info: {name} momentum file not found at {p}")
            continue
        try:
            mom = np.loadtxt(p, delimiter=',')
            mom = np.atleast_2d(mom).astype(float)
        except Exception as e:
            print(f"Failed to read {p}: {e}")
            continue

        tpath = times.get(name)
        t = None
        if tpath and os.path.exists(tpath):
            try:
                t = np.loadtxt(tpath, delimiter=',')
                t = np.atleast_1d(t).astype(float)
            except Exception:
                t = None
        if t is None:
            t = np.arange(mom.shape[0])

        # Ensure orientation: if momentum rows don't match time length
        # try transposing, otherwise trim to the minimum length
        if t is not None:
            if mom.shape[0] != t.shape[0]:
                # common case: file saved transposed (cols=time)
                if mom.shape[1] == t.shape[0]:
                    mom = mom.T
                    print(f"Info: transposed momentum matrix for {name}")
                else:
                    minlen = min(mom.shape[0], t.shape[0])
                    print(f"Warning: shape mismatch for {name}, trimming to {minlen} samples")
                    mom = mom[:minlen, :]
                    t = t[:minlen]

        idx0 = joint_idx - 1
        if idx0 < 0 or idx0 >= mom.shape[1]:
            print(f"{name}: requested joint {joint_idx} out of range (cols={mom.shape[1]})")
            continue

        plt.plot(t, mom[:, idx0], label=name, color=colmap.get(name), linewidth=2)
        any_plotted = True

    if not any_plotted:
        print("No momentum data available for joint comparison.")
        return False

    plt.xlabel('Time [s]')
    plt.ylabel(f'Momentum (joint {joint_idx})')
    plt.title(f'Joint {joint_idx} Momentum Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = f"momentum_compare_joint{joint_idx}_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    return True


def plot_phase_plane(tag, dpi_set, joint_index=6):
    """绘制指定关节（0-based 索引）的相平面图（q vs qdot），对比 rk4/pybullet/etsvi。"""
    print(f"Generating phase plane for joint {joint_index+1}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')

    params = build_params_label()
    csv_dir_etsvi = os.path.join(base, params, 'etsvi')
    csv_dir_rk4   = os.path.join(base, params, 'rk4')
    csv_dir_py    = os.path.join(base, params, 'pybullet')

    # --- RK4: load q_history and v_history if available ---
    q_rk4 = None
    v_rk4 = None
    t_rk4 = None
    try:
        q_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'q_history.csv'), delimiter=',')
        # ensure 2D
        if q_rk4.ndim == 1:
            q_rk4 = q_rk4[:, None]
    except Exception:
        pass

    try:
        v_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'v_history.csv'), delimiter=',')
        if v_rk4.ndim == 1:
            v_rk4 = v_rk4[:, None]
    except Exception:
        v_rk4 = None

    try:
        t_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'time_history.csv'), delimiter=',')
    except Exception:
        t_rk4 = None

    # --- ETSVI: load q_history and time, v estimated by diff ---
    q_etsvi = None
    t_etsvi = None
    v_etsvi = None
    try:
        q_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'q_history.csv'), delimiter=',')
        if q_etsvi.ndim == 1:
            q_etsvi = q_etsvi[:, None]
    except Exception:
        pass
    try:
        t_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
    except Exception:
        t_etsvi = None
    if q_etsvi is not None and t_etsvi is not None:
        # finite difference to estimate velocity for joint
        q_col = q_etsvi[:, joint_index] if q_etsvi.shape[1] > joint_index else None
        if q_col is not None:
            v_etsvi = np.gradient(q_col, t_etsvi)

    # --- PyBullet: load q_history and time, v estimated by diff ---
    q_py = None
    t_py = None
    v_py = None
    try:
        q_py = np.loadtxt(os.path.join(csv_dir_py, 'q_history.csv'), delimiter=',')
        if q_py.ndim == 1:
            q_py = q_py[:, None]
    except Exception:
        pass
    try:
        t_py = np.loadtxt(os.path.join(csv_dir_py, 'time_history.csv'), delimiter=',')
    except Exception:
        t_py = None
    if q_py is not None and t_py is not None:
        q_col = q_py[:, joint_index] if q_py.shape[1] > joint_index else None
        if q_col is not None:
            v_py = np.gradient(q_col, t_py)

    # --- Prepare figure ---
    _init_fig(figsize=(8, 6))

    plotted = False
    # plot rk4
    if q_rk4 is not None:
        qcol = q_rk4[:, joint_index] if q_rk4.shape[1] > joint_index else None
        if qcol is not None:
            if v_rk4 is not None and v_rk4.shape[1] > joint_index:
                vcol = v_rk4[:, joint_index]
            elif t_rk4 is not None:
                vcol = np.gradient(qcol, t_rk4)
            else:
                vcol = np.gradient(qcol)
            plt.plot(qcol, vcol, label='RK4', linewidth=1)
            plotted = True

    # plot etsvi
    if q_etsvi is not None and v_etsvi is not None:
        plt.plot(q_etsvi[:, joint_index], v_etsvi, label='ETSVI', linewidth=1)
        plotted = True

    # plot pybullet
    if q_py is not None and v_py is not None:
        plt.plot(q_py[:, joint_index], v_py, label='PyBullet', linewidth=1)
        plotted = True

    if not plotted:
        print('No data available to plot phase plane for joint', joint_index+1)
        return False

    plt.xlabel(f'Joint {joint_index+1} Position [rad]')
    plt.ylabel(f'Joint {joint_index+1} Velocity [rad/s]')
    # make labels consistent with ETSVI plotting (q / qdot) and enforce 1:1 aspect
    plt.xlabel(f'Joint {joint_index+1} q [rad]')
    plt.ylabel(f'Joint {joint_index+1} qdot [rad/s]')
    plt.title(f'Phase Plane - Joint {joint_index+1}')
    try:
        plt.gca().set_aspect('equal', adjustable='box')
    except Exception:
        pass
    plt.grid(True)
    plt.legend()
    filename = f'phase_plane_joint{joint_index+1}_{tag}.png'
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=True)

    return True


def plot_rk4_pybullet_joint_phase_separate(tag, dpi_set, joint_idx=2):
    """分别为 RK4 与 PyBullet 绘制指定关节（1-based 编号）的相图（q vs qdot），并保存各自图像。"""
    print(f"Generating separate phase plots for joint {joint_idx} (RK4 & PyBullet)...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params = build_params_label()

    csv_rk4 = os.path.join(base, params, 'rk4')
    csv_py = os.path.join(base, params, 'pybullet')

    target_idx = joint_idx - 1

    # --- RK4 ---
    q_rk4 = None
    v_rk4 = None
    t_rk4 = None
    try:
        q_rk4 = np.loadtxt(os.path.join(csv_rk4, 'q_history.csv'), delimiter=',')
        if q_rk4.ndim == 1:
            q_rk4 = q_rk4[:, None]
    except Exception:
        q_rk4 = None
    try:
        v_rk4 = np.loadtxt(os.path.join(csv_rk4, 'v_history.csv'), delimiter=',')
        if v_rk4.ndim == 1:
            v_rk4 = v_rk4[:, None]
    except Exception:
        v_rk4 = None
    try:
        t_rk4 = np.loadtxt(os.path.join(csv_rk4, 'time_history.csv'), delimiter=',')
    except Exception:
        t_rk4 = None

    # 绘制 RK4 相图
    if q_rk4 is not None:
        if target_idx < 0 or target_idx >= q_rk4.shape[1]:
            print(f"RK4: joint {joint_idx} out of range")
        else:
            qcol = q_rk4[:, target_idx]
            if v_rk4 is not None and v_rk4.shape[1] > target_idx:
                vcol = v_rk4[:, target_idx]
            elif t_rk4 is not None:
                vcol = np.gradient(qcol, t_rk4)
            else:
                vcol = np.gradient(qcol)
            _init_fig(figsize=(7,6))
            plt.plot(qcol, vcol, label='RK4', linewidth=1, color='tab:blue')
            # match ETSVI labels and enforce 1:1
            plt.xlabel(f'Joint {joint_idx} q [rad]')
            plt.ylabel(f'Joint {joint_idx} qdot [rad/s]')
            plt.title(f'RK4 Phase Portrait')
            try:
                plt.gca().set_aspect('equal', adjustable='box')
            except Exception:
                pass
            plt.grid(True)
            plt.legend()
            filename = f'phase_rk4_joint{joint_idx}_{tag}.png'
            _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    else:
        print('RK4 q_history.csv not found or unreadable; skipping RK4 phase plot')

    # --- PyBullet ---
    q_py = None
    t_py = None
    try:
        q_py = np.loadtxt(os.path.join(csv_py, 'q_history.csv'), delimiter=',')
        if q_py.ndim == 1:
            q_py = q_py[:, None]
    except Exception:
        q_py = None
    try:
        t_py = np.loadtxt(os.path.join(csv_py, 'time_history.csv'), delimiter=',')
    except Exception:
        t_py = None

    if q_py is not None:
        if target_idx < 0 or target_idx >= q_py.shape[1]:
            print(f"PyBullet: joint {joint_idx} out of range")
        else:
            qcol = q_py[:, target_idx]
            if t_py is not None:
                vcol = np.gradient(qcol, t_py)
            else:
                vcol = np.gradient(qcol)
            _init_fig(figsize=(7,6))
            plt.plot(qcol, vcol, label='PyBullet', linewidth=1, color='tab:orange')
            # match ETSVI labels and enforce 1:1
            plt.xlabel(f'Joint {joint_idx} q [rad]')
            plt.ylabel(f'Joint {joint_idx} qdot [rad/s]')
            plt.title(f'PyBullet Phase Portrait')
            try:
                plt.gca().set_aspect('equal', adjustable='box')
            except Exception:
                pass
            plt.grid(True)
            plt.legend()
            filename = f'phase_pybullet_joint{joint_idx}_{tag}.png'
            _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    else:
        print('PyBullet q_history.csv not found or unreadable; skipping PyBullet phase plot')

    return True


def plot_rk4_phase(tag, dpi_set, joint_idx=2):
    print(f"Generating RK4 phase portrait for joint {joint_idx}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label(include_lyap=True)
    csv_dir = os.path.join(base, params_base, 'rk4')
    q_path = os.path.join(csv_dir, 'q_history.csv')
    time_path = os.path.join(csv_dir, 'time_history.csv')

    if not os.path.exists(q_path):
        print(f"No RK4 q_history found at {q_path}; skipping phase plot.")
        return None

    try:
        q_hist = np.loadtxt(q_path, delimiter=',')
        q_hist = np.atleast_2d(q_hist).astype(float)
    except Exception as e:
        print(f"Failed to read {q_path}: {e}")
        return None

    try:
        t = np.loadtxt(time_path, delimiter=',')
        t = np.atleast_1d(t).astype(float)
    except Exception:
        t = np.arange(q_hist.shape[0])

    idx0 = joint_idx - 1
    if idx0 < 0 or idx0 >= q_hist.shape[1]:
        print(f"Requested joint index {joint_idx} out of range (available columns={q_hist.shape[1]})")
        return None

    # 使用 np.gradient 保留与 q_hist 相同长度
    try:
        qdot = np.gradient(q_hist[:, idx0], t)
    except Exception:
        # fallback numerical diff
        dt = t[1] - t[0] if t.size > 1 else 1.0
        qdot = np.zeros(q_hist.shape[0])
        qdot[1:] = np.diff(q_hist[:, idx0]) / dt

    _init_fig(figsize=(6, 6))
    plt.plot(q_hist[:, idx0], qdot, color='#1F77B4', linewidth=1.5)
    plt.scatter(q_hist[0, idx0], qdot[0], marker='o', color='green', label='start', s=100)
    plt.scatter(q_hist[-1, idx0], qdot[-1], marker='X', color='red', label='end', s=100)
    plt.xlabel(f'Joint {joint_idx} q [rad]')
    plt.ylabel(f'Joint {joint_idx} qdot [rad/s]')
    plt.title(f'RK4 Phase Portrait')
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-15, 15)
    handles, leg_labels = plt.gca().get_legend_handles_labels()
    if leg_labels:
        plt.legend()
    filename = f"phase_rk4_joint{joint_idx}_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    return True

def plot_pybullet_phase(tag, dpi_set, joint_idx=2):
    print(f"Generating PyBullet phase portrait for joint {joint_idx}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label(include_lyap=True)
    csv_dir = os.path.join(base, params_base, 'pybullet')
    q_path = os.path.join(csv_dir, 'q_history.csv')
    time_path = os.path.join(csv_dir, 'time_history.csv')

    if not os.path.exists(q_path):
        print(f"No PyBullet q_history found at {q_path}; skipping phase plot.")
        return None

    try:
        q_hist = np.loadtxt(q_path, delimiter=',')
        q_hist = np.atleast_2d(q_hist).astype(float)
    except Exception as e:
        print(f"Failed to read {q_path}: {e}")
        return None

    try:
        t = np.loadtxt(time_path, delimiter=',')
        t = np.atleast_1d(t).astype(float)
    except Exception:
        t = np.arange(q_hist.shape[0])

    idx0 = joint_idx - 1
    if idx0 < 0 or idx0 >= q_hist.shape[1]:
        print(f"Requested joint index {joint_idx} out of range (available columns={q_hist.shape[1]})")
        return None

    # 使用 np.gradient 保留与 q_hist 相同长度
    try:
        qdot = np.gradient(q_hist[:, idx0], t)
    except Exception:
        # fallback numerical diff
        dt = t[1] - t[0] if t.size > 1 else 1.0
        qdot = np.zeros(q_hist.shape[0])
        qdot[1:] = np.diff(q_hist[:, idx0]) / dt

    _init_fig(figsize=(6, 6))
    plt.plot(q_hist[:, idx0], qdot, color='#FF7F0E', linewidth=1.5)
    plt.scatter(q_hist[0, idx0], qdot[0], marker='o', color='green', label='start', s=100)
    plt.scatter(q_hist[-1, idx0], qdot[-1], marker='X', color='red', label='end', s=100)
    plt.xlabel(f'Joint {joint_idx} q [rad]')
    plt.ylabel(f'Joint {joint_idx} qdot [rad/s]')
    plt.title(f'PyBullet Phase Portrait')
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-15, 15)
    handles, leg_labels = plt.gca().get_legend_handles_labels()
    if leg_labels:
        plt.legend()
    filename = f"phase_pybullet_joint{joint_idx}_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    return True

def plot_poincare_section(tag, dpi_set, joint_index=6, trigger_joint_index=0, surface='q=0', direction='+'):
    """绘制关节的庞加莱截面：用 trigger 关节的 q 过零事件作为截面，提取 target 关节的 (q,v)。
    - joint_index: 目标关节（默认关节7，0-based）
    - trigger_joint_index: 触发过零检测的关节（默认关节1，0-based）
    - direction: '+' 表示从负到正穿越，'-' 表示正到负，其它表示任意过零
    对 ETSVI / PyBullet 无速度记录时用时间序列差分估计速度。
    """
    print(f"Generating Poincaré section for joint {joint_index+1}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    csv_dir_etsvi = os.path.join(base, 'etsvi')
    csv_dir_rk4   = os.path.join(base, 'rk4')
    csv_dir_py    = os.path.join(base, 'pybullet')

    def _load_q_v_t(csv_dir):
        q = v = t = None
        try:
            q = np.loadtxt(os.path.join(csv_dir, 'q_history.csv'), delimiter=',')
            if q.ndim == 1:
                q = q[:, None]
        except Exception:
            q = None
        try:
            v = np.loadtxt(os.path.join(csv_dir, 'v_history.csv'), delimiter=',')
            if v.ndim == 1:
                v = v[:, None]
        except Exception:
            v = None
        try:
            t = np.loadtxt(os.path.join(csv_dir, 'time_history.csv'), delimiter=',')
        except Exception:
            t = None
        if q is not None and v is None:
            # fallback: estimate v by gradient if time available
            if t is not None and q.shape[0] == t.shape[0]:
                v_est = np.gradient(q[:, joint_index] if q.shape[1] > joint_index else q[:, 0], t)
                v = np.zeros_like(q)
                v[:, joint_index if q.shape[1] > joint_index else 0] = v_est
            else:
                v = None
        return q, v, t

    def _extract_poincare(q_trigger, q_target, v_target, tarr):
        if q_trigger is None or q_target is None or v_target is None:
            return []
        if tarr is None:
            # assume unit time step if not provided
            tarr = np.arange(len(q_target))
        pts = []
        for i in range(1, len(q_target)):
            qt0, qt1 = q_trigger[i-1], q_trigger[i]
            q0, q1 = q_target[i-1], q_target[i]
            v0, v1 = v_target[i-1], v_target[i]
            if direction == '+':
                crossing = (qt0 <= 0 and qt1 >= 0)
            elif direction == '-':
                crossing = (qt0 >= 0 and qt1 <= 0)
            else:
                crossing = (qt0 * qt1 <= 0)
            if not crossing:
                continue
            dq = qt1 - qt0
            dt = tarr[i] - tarr[i-1]
            if abs(dq) < 1e-12:
                alpha = 0.0
            else:
                alpha = -qt0 / dq
            alpha = np.clip(alpha, 0.0, 1.0)
            v_cross = v0 + alpha * (v1 - v0)
            t_cross = tarr[i-1] + alpha * dt
            # section at q=0
            q_cross = q0 + alpha * (q1 - q0)
            pts.append((q_cross, v_cross, t_cross))
        return pts

    data = {}
    for name, csv_dir in [('RK4', csv_dir_rk4), ('ETSVI', csv_dir_etsvi), ('PyBullet', csv_dir_py)]:
        q, v, t = _load_q_v_t(csv_dir)
        if q is not None:
            col_target = joint_index if q.shape[1] > joint_index else None
            q_target = q[:, col_target] if col_target is not None else None
            col_trig = trigger_joint_index if q.shape[1] > trigger_joint_index else None
            q_trig = q[:, col_trig] if col_trig is not None else None
        else:
            q_target = None
            q_trig = None
        if v is not None:
            col_target_v = joint_index if v.shape[1] > joint_index else None
            v_target = v[:, col_target_v] if col_target_v is not None else None
        else:
            v_target = None
        pts = _extract_poincare(q_trig, q_target, v_target, t)
        data[name] = pts

    _init_fig(figsize=(7, 6))
    cmap = plt.get_cmap('tab10')
    colors = {
        'RK4': cmap(1),
        'ETSVI': cmap(0),
        'PyBullet': cmap(2)
    }
    markers = {
        'RK4': 'o',
        'ETSVI': 's',
        'PyBullet': '^'
    }

    plotted = False
    for name, pts in data.items():
        if pts:
            q_vals = [p[0] for p in pts]
            v_vals = [p[1] for p in pts]
            plt.scatter(q_vals, v_vals, label=name, s=18, marker=markers.get(name, 'o'), alpha=0.8, color=colors.get(name, None))
            plotted = True

    if not plotted:
        print('No Poincaré points found for joint', joint_index+1)
        return False

    plt.xlabel(f'Joint {joint_index+1} Position [rad] (section q=0)')
    plt.ylabel(f'Joint {joint_index+1} Velocity [rad/s]')
    plt.title(f'Poincaré Section - Joint {joint_index+1} (q crossing 0, direction {direction})')
    plt.grid(True, alpha=0.4)
    plt.legend()
    filename = f'poincare_joint{joint_index+1}_{tag}.png'
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=True)
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-sim', action='store_true', help='Skip running simulations and only plot using existing CSVs')
    args = parser.parse_args()

    # 运行仿真（除非用户要求跳过）
    if not args.skip_sim:
        # if not run_etsvi():
        #     return 1

        if not run_rk4():
            return 1

        # 运行 pybullet 对照仿真（内联执行 pybullet_sim.py）
        if not run_pybullet_inline():
            print("Warning: pybullet inline run failed or skipped. Proceeding to plotting with available CSVs.")

    # 绘制结果
    if not plot_results("vs_r_p", 500):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
