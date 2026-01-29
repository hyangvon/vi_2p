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
    # 保存到按参数命名的子文件夹下，按 experiment params / tag 分类
    params_label = build_params_label()
    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi_2p/fig/{params_label}/{tag}")
    os.makedirs(save_dir, exist_ok=True)

    # 尝试从配置文件读取运行参数并加入文件名
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
            # 备用：简单文本搜索
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
        # use defaults if not found
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

# def run_etsvi_op():
#     """运行 C++ 仿真节点"""
#     config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/config/vi_params.yaml')
#
#     print("Starting etsvi_op simulation...")
#
#     try:
#         result = subprocess.run([
#             'ros2', 'run', 'vi_2p', 'etsvi_op_node',
#             '--ros-args', '--params-file', config_file
#         ], check=True)
#
#         print("Simulation completed successfully")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Simulation failed: {e}")
#         if e.stderr:
#             print(f"Error output: {e.stderr}")
#         return False

def plot_runtime_comparison(tag, dpi_set):
    """绘制三种算法的平均运行时间对比图"""
    print("Generating runtime comparison plot...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label()

    # 定义算法和对应的文件路径
    algorithms = {
        'ctsvi_ad': os.path.join(base, params_base, 'ctsvi_ad', 'avg_runtime.txt'),
        'atsvi_ad': os.path.join(base, params_base, 'atsvi_ad', 'avg_runtime.txt'),
        'etsvi': os.path.join(base, params_base, 'etsvi', 'avg_runtime.txt')
    }

    avg_times = []
    labels = []

    for alg_name, alg_path in algorithms.items():
        try:
            with open(alg_path, 'r') as f:
                avg_time = float(f.read().strip())
                avg_times.append(avg_time)
                labels.append(alg_name.upper())
        except FileNotFoundError:
            print(f"Warning: {alg_path} not found, using default value 0")
            avg_times.append(0)
            labels.append(alg_name.upper())

    # 绘制柱状图（使用统一风格）
    _init_fig(figsize=(10, 6))
    bars = plt.bar(labels, avg_times,
                   color=['skyblue', 'lightgreen', 'lightcoral'],
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


def plot_runtime_vs_energy(tag, dpi_set, csv_paths=None, font_sizes=None):
    """
    绘制平均运行时间 (x) vs 平均能量误差 (y) 的对比图。
    函数会按 csv_paths 的顺序为每种算法绘制连线，x 轴为 runtime，y 轴为 mean(|delta_energy|)。
    """

    print("Generating runtime vs energy plot...")
    # font_sizes: dict with keys 'title','label','tick','legend','text'
    if font_sizes is None:
        font_sizes = {}
    title_fs = font_sizes.get('title', 18)
    label_fs = font_sizes.get('label', 16)
    tick_fs = font_sizes.get('tick', 15)
    legend_fs = font_sizes.get('legend', 16)
    text_fs = font_sizes.get('text', 15)
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')

    params_base = [
        'q0p2_dt0p01_T40_a0p4_b0p04', 
        'q0p2_dt0p0075_T40_a0p4_b0p04', 
        'q0p2_dt0p005_T40_a0p4_b0p04', 
        # 'q0p2_dt0p0025_T40_a0p4_b0p04', 
        # 'q0p2_dt0p001_T40_a0p4_b0p04', 
        
        'q0p4_dt0p01_T40_a0p4_b0p04', 
        'q0p4_dt0p0075_T40_a0p4_b0p04', 
        'q0p4_dt0p005_T40_a0p4_b0p04',
        # 'q0p4_dt0p0025_T40_a0p4_b0p04', 
        # 'q0p4_dt0p001_T40_a0p4_b0p04', 
    ]

    # 自动根据 params_base 列表生成 csv_paths 映射，键为参数文件夹名（例如 q0p2_dt...）
    csv_paths = {}
    for p in params_base:
        csv_paths[p] = {
            'CTSVI': {
                'runtime': os.path.join(base, p, 'ctsvi_ad', 'avg_runtime.txt'),
                'energy': os.path.join(base, p, 'ctsvi_ad', 'delta_energy_history.csv')
            },
            'ATSVI': {
                'runtime': os.path.join(base, p, 'atsvi_ad', 'avg_runtime.txt'),
                'energy': os.path.join(base, p, 'atsvi_ad', 'delta_energy_history.csv')
            },
            'C-ATSVI': {
                'runtime': os.path.join(base, p, 'etsvi', 'avg_runtime.txt'),
                'energy': os.path.join(base, p, 'etsvi', 'delta_energy_history.csv')
            }
        }

    # 默认单组行为（保持向后兼容）
    if csv_paths is None:
        params_base = build_params_label()
        methods = {
            'CTSVI': {
                'runtime': os.path.join(base, params_base, 'ctsvi_ad', 'avg_runtime.txt'),
                'energy': os.path.join(base, params_base, 'ctsvi_ad', 'delta_energy_history.csv')
            },
            'ATSVI': {
                'runtime': os.path.join(base, params_base, 'atsvi_ad', 'avg_runtime.txt'),
                'energy': os.path.join(base, params_base, 'atsvi_ad', 'delta_energy_history.csv')
            },
            'C-ATSVI': {
                'runtime': os.path.join(base, params_base, 'etsvi', 'avg_runtime.txt'),
                'energy': os.path.join(base, params_base, 'etsvi', 'delta_energy_history.csv')
            }
        }
        # compute single-point results
        xs = []
        ys = []
        labels = []
        for name, paths in methods.items():
            rt = float('nan')
            if os.path.exists(paths['runtime']):
                try:
                    with open(paths['runtime'], 'r') as f:
                        rt = float(f.read().strip())
                except Exception:
                    rt = float('nan')
            # 读取 energy
            mae = float('nan')
            if os.path.exists(paths['energy']):
                try:
                    arr = np.loadtxt(paths['energy'], delimiter=',')
                    arr = np.atleast_1d(arr).astype(float)
                    mae = float(np.mean(np.abs(arr)))
                except Exception:
                    mae = float('nan')
            # 读取 h_history 以计算总运行时间 = avg_runtime * (n-1)
            total_rt = rt
            try:
                # 首先尝试本算法目录下的 h_history.csv
                h_path = os.path.join(os.path.dirname(paths['runtime']), 'h_history.csv')
                if not os.path.exists(h_path):
                    # 如果是 CTSVI，尝试使用同参数下的 ATSVI 的 h_history.csv 作为备用
                    parent = os.path.dirname(os.path.dirname(paths['runtime']))
                    alt = os.path.join(parent, 'atsvi_ad', 'h_history.csv')
                    if os.path.exists(alt):
                        h_path = alt

                if os.path.exists(h_path):
                    harr = np.loadtxt(h_path, delimiter=',')
                    harr = np.atleast_1d(harr)
                    n_steps = harr.size if harr.ndim == 1 else harr.shape[0]
                    if not np.isnan(rt) and n_steps > 1:
                        total_rt = float(rt) * float(n_steps - 1)
            except Exception:
                total_rt = rt

            xs.append(total_rt)
            ys.append(mae)
            labels.append(name)

        _init_fig(figsize=(6, 6))
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(labels))]
        # 不同算法使用不同 marker
        markers = ['o', 's', '^', 'D', 'v']
        for i, (x, y, lbl, c) in enumerate(zip(xs, ys, labels, colors)):
            m = markers[i % len(markers)]
            plt.scatter(x, y, label=lbl, color=c, s=120, marker=m)
            plt.text(x, y, f' {lbl}', verticalalignment='center', fontsize=text_fs)
        # 仅当存在带标签的 artist 时显示图例，避免警告
        handles, leg_labels = plt.gca().get_legend_handles_labels()
        if leg_labels:
            plt.legend(fontsize=legend_fs)
        plt.xlim(0, 65)
        plt.ylim(0, 0.01)

    else:
        # csv_paths provided: 可以是 dict (param_label -> {alg: {runtime,energy}})
        # or a list of tuples (param_label, mapping)
        # Normalize to list of (param_label, mapping)
        items = None
        if isinstance(csv_paths, dict):
            items = list(csv_paths.items())
        elif isinstance(csv_paths, list) or hasattr(csv_paths, '__iter__'):
            items = list(csv_paths)
        else:
            plt.xlabel('Total Runtime (ms)')

        # collect per-algorithm series
        algs = []
        # determine algorithm names from first mapping
        if items:
            first_map = items[0][1]
            algs = list(first_map.keys())

        series = {alg: {'rts': [], 'maes': [], 'labels': []} for alg in algs}
        for param_label, mapping in items:
            for alg in algs:
                rt = float('nan')
                mae = float('nan')
                paths = mapping.get(alg, {})
                if 'runtime' in paths and os.path.exists(paths['runtime']):
                    try:
                        with open(paths['runtime'], 'r') as f:
                            rt = float(f.read().strip())
                    except Exception:
                        rt = float('nan')
                # 将 avg runtime 转为 total runtime: avg * (n_steps - 1) if h_history exists
                try:
                    h_path = os.path.join(os.path.dirname(paths.get('runtime', '')) , 'h_history.csv')
                    if not os.path.exists(h_path):
                        parent = os.path.dirname(os.path.dirname(paths.get('runtime', '') or ''))
                        alt = os.path.join(parent, 'atsvi_ad', 'h_history.csv')
                        if os.path.exists(alt):
                            h_path = alt
                    if os.path.exists(h_path):
                        harr = np.loadtxt(h_path, delimiter=',')
                        harr = np.atleast_1d(harr)
                        n_steps = harr.size if harr.ndim == 1 else harr.shape[0]
                        if not np.isnan(rt) and n_steps > 1:
                            rt = float(rt) * float(n_steps - 1)
                except Exception:
                    pass
                if 'energy' in paths and os.path.exists(paths['energy']):
                    try:
                        arr = np.loadtxt(paths['energy'], delimiter=',')
                        arr = np.atleast_1d(arr).astype(float)
                        mae = float(np.mean(np.abs(arr)))
                    except Exception:
                        mae = float('nan')
                series[alg]['rts'].append(rt)
                series[alg]['maes'].append(mae)
                series[alg]['labels'].append(param_label)

        # 绘制折线：每个算法分别为 q0p2 和 q0p4 两组绘制一条线（不同线型）
        _init_fig(figsize=(6, 5.5))
        cmap = plt.get_cmap('tab10')
        colors = {alg: cmap(i) for i, alg in enumerate(series.keys())}
        # 为不同算法分配不同 marker
        marker_list = ['o', '^', 'D', 'v', 'P', '*', 's']
        marker_list_2 = ['D', 'v', 'P', '*', 's', 'o', '^']
        for i, (alg, data) in enumerate(series.items()):
            rts = np.array(data['rts'])
            maes = np.array(data['maes'])
            lbls = data['labels']
            m = marker_list[i % len(marker_list)]
            m2 = marker_list_2[i % len(marker_list_2)]
            # 按 q_init 分组 (q0p2 / q0p4)
            mask_q02 = [str(lbl).startswith('q0p2') for lbl in lbls]
            mask_q04 = [str(lbl).startswith('q0p4') for lbl in lbls]
            # 提取子序列
            rts_q02 = rts[np.array(mask_q02, dtype=bool)]
            maes_q02 = maes[np.array(mask_q02, dtype=bool)]
            rts_q04 = rts[np.array(mask_q04, dtype=bool)]
            maes_q04 = maes[np.array(mask_q04, dtype=bool)]

            # print("rts_q02.size", rts_q02.size)
            # print("rts_q04.size", rts_q04.size)

            if rts_q02.size:
                plt.plot(rts_q02, maes_q02, marker=m, linestyle='-', label=f'{alg} q0.2', color=colors.get(alg))
            if rts_q04.size:
                plt.plot(rts_q04, maes_q04, marker=m2, linestyle='--', label=f'{alg} q0.4', color=colors.get(alg))

        # 仅当存在带标签的 artist 时显示图例，避免警告
        handles, leg_labels = plt.gca().get_legend_handles_labels()
        if leg_labels:
            plt.legend(fontsize=legend_fs)
        # plt.xlim(0, 65)
        # plt.ylim(-0.002, 0.01)

    # plt.xlabel('Average Runtime (ms)')
    plt.xlabel('Total Runtime (ms)', fontsize=label_fs)
    plt.ylabel('Mean Absolute Energy Error (J)', fontsize=label_fs)
    plt.title('Runtime vs Energy Error', fontsize=title_fs)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=tick_fs)
    plt.yticks(fontsize=tick_fs)

    # 在图中添加三条平行注释箭头，指向左下角区域，表示越靠近左下角越好
    # try:
    #     target_x, target_y = 0.2, 0.2  # 目标点（axes fraction）
    #     # 三条箭头的起始文本位置（在图上方略有垂直间隔）
    #     text_positions = [(0.4, 0.4), (0.45, 0.36), (0.5, 0.32)]
    #     arrow_props = dict(arrowstyle='->', color='black', lw=1.2)
    #     for i, tp in enumerate(text_positions):
    #         # 每条箭头指向目标点并略微错开目标 y 以产生“簇”效果
    #         plt.annotate('', xy=(target_x + i*0.05, target_y - i*0.04), xycoords='axes fraction',
    #                      xytext=tp, textcoords='axes fraction', arrowprops=arrow_props)

    #     # 旁注文字（仅一处描述）
    #     plt.text(0.62, 0.50, 'Direction of higher\nenergy accuracy and\nlower runtime',
    #              transform=plt.gca().transAxes, fontsize=11,
    #              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.9))
    # except Exception:
    #     pass

    filename = f"runtime_vs_energy_{tag}.png"
    # _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi_2p/fig/vs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi_set)
    print(f"Saved: {save_path}")


def compute_and_save_energy_errors(tag):
    """计算 CTSVI/ATSVI/ETSVI 的能量误差平均值并保存为 CSV，同时打印到终端。
        输出保存到: src/vi_2p/csv/<tag>/energy_error_summary_<tag>.csv
    """
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label()
    methods = {
        'CTSVI_AD': os.path.join(base, params_base, 'ctsvi_ad', 'delta_energy_history.csv'),
        'ATSVI_AD': os.path.join(base, params_base, 'atsvi_ad', 'delta_energy_history.csv'),
        'ETSVI': os.path.join(base, params_base, 'etsvi', 'delta_energy_history.csv'),
    }

    results = []
    for name, path in methods.items():
        if os.path.exists(path):
            try:
                arr = np.loadtxt(path, delimiter=',')
                arr = np.atleast_1d(arr).astype(float)
                mean_abs = float(np.mean(np.abs(arr)))
                rms = float(np.sqrt(np.mean(arr**2)))
                n = int(arr.size)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                mean_abs = float('nan')
                rms = float('nan')
                n = 0
        else:
            mean_abs = float('nan')
            rms = float('nan')
            n = 0
        results.append((name, mean_abs, rms, n))

    # 打印到终端
    print("Energy error summary:")
    for name, mean_abs, rms, n in results:
        print(f"- {name}: samples={n}, mean_abs_error={mean_abs:.6e}, rms={rms:.6e}")

    # 保存 CSV 到对应参数子文件夹下（包含 lyap）
    out_dir = os.path.join(base, params_base)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'energy_error_summary_{tag}.csv')
    with open(out_path, 'w') as f:
        f.write('method,mean_abs_error,rms_error,samples\n')
        for name, mean_abs, rms, n in results:
            f.write(f"{name},{mean_abs},{rms},{n}\n")

    print(f"Saved energy error summary to {out_path}")
    return out_path


def plot_momentum(tag, dpi_set):
    """
    绘制 CTSVI / ATSVI / C-ATSVI 的动量历史（momentum_history.csv）。
    如果文件存在则读取并绘制每个分量随时间的变化。
    """
    print("Generating momentum plot...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label(include_lyap=True)

    csv_dirs = {
        'CTSVI': os.path.join(base, params_base, 'ctsvi_ad'),
        'ATSVI': os.path.join(base, params_base, 'atsvi_ad'),
        'C-ATSVI': os.path.join(base, params_base, 'etsvi'),
    }

    data = {}
    times = {}
    for name, d in csv_dirs.items():
        mom_path = os.path.join(d, 'momentum_history.csv')
        time_path = os.path.join(d, 'time_history.csv')
        if os.path.exists(mom_path):
            try:
                arr = np.loadtxt(mom_path, delimiter=',')
                arr = np.atleast_2d(arr).astype(float)
                # ensure shape (N, D)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                data[name] = arr
            except Exception as e:
                print(f"Warning: failed to read {mom_path}: {e}")
                data[name] = None
        else:
            print(f"Info: {mom_path} not found")
            data[name] = None

        if os.path.exists(time_path):
            try:
                t = np.loadtxt(time_path, delimiter=',')
                t = np.atleast_1d(t).astype(float)
                times[name] = t
            except Exception:
                times[name] = None
        else:
            times[name] = None

    # 如果全部为空则跳过
    if all(v is None for v in data.values()):
        print("No momentum data found for any algorithm; skipping momentum plot.")
        return None

    _init_fig(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    for i, (name, arr) in enumerate(data.items()):
        if arr is None:
            continue
        t = times.get(name)
        if t is None or t.size != arr.shape[0]:
            # fallback to sample index
            t = np.arange(arr.shape[0])

        # plot each column as a separate line
        for j in range(arr.shape[1]):
            label = f"{name} m{j+1}" if arr.shape[1] > 1 else f"{name}"
            plt.plot(t, arr[:, j], label=label, color=cmap(i), linewidth=1.5)

    plt.xlabel('Time [s]')
    plt.ylabel('Momentum')
    plt.title('Momentum History (CTSVI / ATSVI / C-ATSVI)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = f"momentum_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    return True


def plot_etsvi_joint(tag, dpi_set, joint_idx=2):
    """
    仅绘制 ETSVI 的指定关节（1-based 索引）的通用动量随时间变化。
    默认绘制关节7（joint_idx=7）。
    """
    print(f"Generating ETSVI joint {joint_idx} momentum plot...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label(include_lyap=True)
    csv_dir = os.path.join(base, params_base, 'etsvi')

    mom_path = os.path.join(csv_dir, 'momentum_history.csv')
    time_path = os.path.join(csv_dir, 'time_history.csv')

    if not os.path.exists(mom_path):
        print(f"No ETSVI momentum file found at {mom_path}")
        return None

    try:
        mom = np.loadtxt(mom_path, delimiter=',')
        mom = np.atleast_2d(mom).astype(float)
    except Exception as e:
        print(f"Failed to read {mom_path}: {e}")
        return None

    try:
        t = np.loadtxt(time_path, delimiter=',')
        t = np.atleast_1d(t).astype(float)
    except Exception:
        t = np.arange(mom.shape[0])

    idx0 = joint_idx - 1
    idx0 = joint_idx - 1
    # If at least two columns available, plot joint1, joint2 and total momentum
    n_cols = mom.shape[1]
    _init_fig(figsize=(6,4))
    if n_cols >= 2:
        p1 = mom[:, 0]
        p2 = mom[:, 1]
        p_total = np.sum(mom, axis=1)
        plt.plot(t, p1, label='joint 1', color='C0', linewidth=1.5)
        plt.plot(t, p2, label='joint 2', color='C1', linewidth=1.5)
        plt.plot(t, p_total, label='total', color='C2', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Angular momentum')
        plt.title('ETSVI Joint 1 & 2 and Total Momentum')
        plt.grid(True, alpha=0.3)
        plt.legend()
        filename = f"momentum_etsvi_joints12_{tag}.png"
        _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
        return True
    else:
        # fallback to original single-joint plot
        if idx0 < 0 or idx0 >= n_cols:
            print(f"Requested joint index {joint_idx} out of range (available columns={n_cols})")
            return None
        plt.plot(t, mom[:, idx0], color='C2', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel(f'Momentum (joint {joint_idx})')
        plt.title(f'ETSVI Joint {joint_idx} Momentum')
        plt.grid(True, alpha=0.3)
        filename = f"momentum_etsvi_joint{joint_idx}_{tag}.png"
        _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
        return True


def plot_etsvi_phase(tag, dpi_set, joint_idx=2):
    """
    绘制 ETSVI 指定关节的相图：q (角度) vs qdot (角速度)。
    joint_idx 为 1-based 索引，默认第 2 关节。
    """
    print(f"Generating ETSVI phase portrait for joint {joint_idx}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label(include_lyap=True)
    csv_dir = os.path.join(base, params_base, 'etsvi')

    q_path = os.path.join(csv_dir, 'q_history.csv')
    time_path = os.path.join(csv_dir, 'time_history.csv')

    if not os.path.exists(q_path):
        print(f"No ETSVI q_history found at {q_path}; skipping phase plot.")
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
    plt.plot(q_hist[:, idx0], qdot, color='#D62728', linewidth=1.5)
    plt.scatter(q_hist[0, idx0], qdot[0], marker='o', color='green', label='start', s=100)
    plt.scatter(q_hist[-1, idx0], qdot[-1], marker='X', color='red', label='end', s=100)
    plt.xlabel(f'Joint {joint_idx} q [rad]')
    plt.ylabel(f'Joint {joint_idx} qdot [rad/s]')
    plt.title(f'C-ATSVI Phase Portrait')
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-15, 15)
    handles, leg_labels = plt.gca().get_legend_handles_labels()
    if leg_labels:
        plt.legend()
    filename = f"phase_etsvi_joint{joint_idx}_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    return True

def plot_7dof_pendulum():
    # 参数设置
    n_links = 7
    link_length = 1.0  # 根据 URDF 文件，joint 到 joint 距离为 1.0m
    
    # 初始关节角 (每个关节都是 0.2 rad)
    q = np.array([0.2] * n_links)
    
    # 运动学正解 (Forward Kinematics) - 仅计算 X-Z 平面
    # 假设基座在 (0, 0)
    x = [0.0]
    z = [0.0]
    
    current_angle = 0.0
    
    # 循环计算每个关节的位置
    for i in range(n_links):
        # 累加角度（相对角度 -> 绝对角度）
        # 0位姿是竖直向下 (-Z)，绕 Y 轴正向旋转会将连杆向 +X 方向摆动
        current_angle += q[i]
        
        # 计算下一个关节的位置
        # x = L * sin(theta)
        # z = -L * cos(theta) (因为基准是向下)
        next_x = x[-1] + link_length * np.sin(current_angle)
        next_z = z[-1] - link_length * np.cos(current_angle)
        
        x.append(next_x)
        z.append(next_z)
        
    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 1. 画连杆 (实线)
    ax.plot(x, z, 'o-', linewidth=3, color='#34495e', markersize=8, zorder=1, label='Links')
    
    # 2. 画关节 (红色圆点)
    ax.scatter(x[1:-1], z[1:-1], s=100, c='#e74c3c', zorder=2, label='Joints')
    
    # 3. 画基座 (固定点)
    ax.scatter(x[0], z[0], s=200, marker='^', c='black', zorder=3, label='Base (Fixed)')
    
    # 4. 画末端执行器 (TCP)
    ax.scatter(x[-1], z[-1], s=150, c='#2ecc71', marker='*', zorder=3, label='End Effector')
    
    # 5. 画零位参考线 (虚线)
    ax.plot([0, 0], [0, -n_links*link_length], '--', color='gray', alpha=0.5, label='Zero Configuration')

    # 标注和美化
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(f'7-DoF Pendulum Initial Configuration\n($q_i = 0.2$ rad for all joints)', fontsize=14)
    ax.legend()
    
    # 添加角度累积的注释
    ax.text(x[3]+0.2, z[3], "Cumulative Curvature", fontsize=10, color='#34495e', style='italic')

    # 调整视野
    plt.tight_layout()
    plt.show()

def plot_results(tag, dpi_set):
    """绘制仿真结果"""
    print("Generating plots...")

    # ---------- 1. 读取数据 ----------
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/csv/')
    params_base = build_params_label(include_lyap=True)

    csv_dir_ctsvi_ad = os.path.join(base, params_base, 'ctsvi_ad')
    csv_dir_atsvi_ad = os.path.join(base, params_base, 'atsvi_ad')
    csv_dir_etsvi    = os.path.join(base, params_base, 'etsvi')

    print(f"{csv_dir_ctsvi_ad}")

    if not os.path.exists(os.path.join(csv_dir_atsvi_ad, 'q_history.csv')):
        print("CSV files not found. Simulation may have failed.")
        return False

    try:
        # q_history = np.loadtxt(os.path.join(csv_dir, 'q_history.csv'), delimiter=',')
        # if q_history.ndim == 1:
        #     q_history = q_history.reshape(-1, 1)
        tcp_ctsvi = np.loadtxt(os.path.join(csv_dir_ctsvi_ad, 'ee_history.csv'), delimiter=',')
        tcp_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'ee_history.csv'), delimiter=',')
        tcp_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'ee_history.csv'), delimiter=',')
        # energy = np.loadtxt(os.path.join(csv_dir, 'energy_history.csv'), delimiter=',')
        delta_energy_ctsvi = np.loadtxt(os.path.join(csv_dir_ctsvi_ad, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'delta_energy_history.csv'), delimiter=',')
        time_ctsvi = np.loadtxt(os.path.join(csv_dir_ctsvi_ad, 'time_history.csv'), delimiter=',')
        time_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'time_history.csv'), delimiter=',')
        time_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
        # momentum = np.loadtxt(os.path.join(csv_dir, 'momentum_history.csv'), delimiter=',')
        step_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'h_history.csv'), delimiter=',')
        step_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'h_history.csv'), delimiter=',')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return False

    # print("q_history shape:", q_history.shape)

    # 自动检测自由度数
    # nq = q_history.shape[1]
    # print(f"Loaded q_history with {nq} DOFs, {q_history.shape[0]} time steps")

    # 创建图形窗口
    # plt.ion()  # 开启交互模式

    # 输出目录将在保存时由 _save_fig 创建

    # # ---------- 2. 绘制关节角随时间 ----------
    # plt.figure(figsize=(10, 5))
    # for i in range(nq):
    #     plt.plot(time, q_history[0:, i], label=f'q{i+1}')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Joint angle [rad]')
    # plt.title('Joint trajectories')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # filename = f"q_{tag}.png"
    # save_path = os.path.join(save_dir, filename)
    # plt.savefig(save_path, dpi = dpi_set)
    # print("Saved:", save_path)
    # # plt.show()
    #
    # ---------- 3. 绘制能量曲线 ----------
    _init_fig()
    # plt.plot(time, energy, label='Total Energy')

    # 使用调色板和更明显的样式以增强可区分度
    cmap = plt.get_cmap('tab10')
    c_ctsvi = cmap(0)
    c_atsvi = cmap(1)
    c_etsvi = cmap(2)


    # CTSVI
    plt.plot(time_ctsvi, delta_energy_ctsvi, label='ΔEnergy of CTSVI', color='#9467BD', linestyle=':', linewidth=1.5)

    # ATSVI
    plt.plot(time_atsvi, delta_energy_atsvi, label='ΔEnergy of ATSVI', color='#000000', linestyle='-', linewidth=1.2)

    # ETSVI
    plt.plot(time_etsvi, delta_energy_etsvi, label='ΔEnergy of C-ATSVI', color='#D62728', linestyle='-', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy evolution')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-0.015, 0.015)
    filename = f"energy_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    #
    # # ---------- 4. 相平面图（q vs qdot） ----------
    # dt = time[1] - time[0]
    # qdot = np.diff(q_history, axis=0) / dt  # numerical derivative
    # plt.figure(figsize=(6, 6))
    # for i in range(nq):
    #     plt.plot(q_history[:-1, i], qdot[:, i], label=f'Joint {i+1}')
    # plt.xlabel('q [rad]')
    # plt.ylabel('qdot [rad/s]')
    # plt.title('Phase Portraits')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # filename = f"phase_{tag}.png"
    # save_path = os.path.join(save_dir, filename)
    # plt.savefig(save_path, dpi = dpi_set)
    # print("Saved:", save_path)
    # # plt.show()

    # ---------- 5. 绘制timestep ----------
    _init_fig()
    plt.plot(time_atsvi, step_atsvi, label='Time Step of ATSVI', linestyle='-.', linewidth=2)
    plt.plot(time_etsvi, step_etsvi, label='Time Step of C-ATSVI', linestyle='--', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Step')
    plt.title('Adaptive Time Step')
    plt.legend()
    plt.grid(True)
    filename = f"step_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    # ---------- 5. position xyz ----------
    _init_fig()
    # CTSVI
    # plt.plot(time_ctsvi, tcp_ctsvi[:, 0], label='px_ctsvi', linestyle='-', linewidth=2)
    plt.plot(time_ctsvi, tcp_ctsvi[:, 2], label='position Z of CTSVI', color=c_ctsvi, linestyle='-', linewidth=2)

    # ATSVI
    # plt.plot(time_atsvi, tcp_atsvi[:, 0], label='px_atsvi', linestyle='-.', linewidth=2)
    plt.plot(time_atsvi, tcp_atsvi[:, 2], label='position Z of ATSVI', color=c_atsvi, linestyle='-.', linewidth=2)
    
    # ETSVI
    # plt.plot(time_etsvi, tcp_etsvi[:, 0], label='px_etsvi', linestyle='--', linewidth=2)
    plt.plot(time_etsvi, tcp_etsvi[:, 2], label='position Z of C-ATSVI', color=c_etsvi, linestyle='--', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Tip Position')
    plt.legend(loc='upper left')
    plt.grid(True)
    # plt.ylim(-8, -3)
    filename = f"tcp_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    # ---------- TCP Z vs Zdot 相平面（仅 ETSVI） ----------
    try:
        z = tcp_etsvi[:, 2]
        # 使用时间差分求速度（与时间数组对齐）
        # 使用 np.gradient 可保持与原数组相同长度
        zdot = np.gradient(z, time_etsvi)

        _init_fig(figsize=(7, 7))
        plt.plot(z, zdot, color=c_etsvi, linestyle='-', linewidth=1.5)
        plt.scatter(z[0], zdot[0], marker='o', color='green', label='start')
        plt.scatter(z[-1], zdot[-1], marker='X', color='red', label='end')
        plt.xlabel('Tip Position Z (m)')
        plt.ylabel('Tip Velocity Z (m/s)')
        plt.title('C-ATSVI Tip Position Phase Plane')
        plt.grid(True, alpha=0.3)
        handles, leg_labels = plt.gca().get_legend_handles_labels()
        if leg_labels:
            plt.legend()
        filename = f"tcp_phase_etsvi_{tag}.png"
        _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
    except Exception as e:
        print(f"Warning: failed to plot ETSVI TCP phase plane: {e}")

    plot_runtime_comparison(tag, dpi_set)

    # 计算并保存三种 VI 方法的能量误差平均值
    compute_and_save_energy_errors(tag)

    # 绘制 平均运行时间 vs 平均能量误差
    plot_runtime_vs_energy(tag, dpi_set)

    # 仅绘制 ETSVI 的关节7 动量（按用户要求）
    try:
        plot_etsvi_joint(tag, dpi_set, joint_idx=2)
        # 绘制第2关节的 q vs qdot 相图
        plot_etsvi_phase(tag, dpi_set, joint_idx=2)
    except Exception as e:
        print(f"Warning: failed to plot ETSVI joint momentum: {e}")

    # print("Plotting completed. Close all plot windows to exit.")

    # 等待用户关闭窗口
    # plt.ioff()  # 关闭交互模式
    # plt.show()  # 阻塞直到所有窗口关闭

    # plot_7dof_pendulum()

    return 1

def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-sim', action='store_true', help='Skip running simulations and only plot using existing CSVs')
    args = parser.parse_args()

    # 运行仿真（除非用户要求跳过）
    if not args.skip_sim:
        if not run_ctsvi():
            return 1

        if not run_atsvi():
            return 1

        if not run_etsvi():
            return 1

    # 绘制结果
    if not plot_results("vs_c_a", 500):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
