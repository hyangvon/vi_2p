#!/usr/bin/env python3
"""
2D 绘制 2 自由度摆（X-Z 平面），参考用户提供的可视化样式
- 零位姿态为竖直向下，初始每个关节角均为 0.2 rad
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# 字体与样式统一设置
FONT_FAMILY = 'DejaVu Sans'
TITLE_FONT_SIZE = 15
LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
TICK_FONT_SIZE = 12
TITLE_FONT_WEIGHT = 'bold'

DEFAULT_DPI = 500

def _apply_style():
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

def compute_fk_2d(q, link_length=1.0):
    """解析计算 2D 正向运动学（X-Z 平面），基座在 (0,0)，向下为负 Z。"""
    n_links = len(q)
    x = [0.0]
    z = [0.0]
    current_angle = 0.0
    for i in range(n_links):
        current_angle += q[i]
        next_x = x[-1] + link_length * np.sin(current_angle)
        next_z = z[-1] - link_length * np.cos(current_angle)
        x.append(next_x)
        z.append(next_z)
    return np.array(x), np.array(z)


def plot_chain(x, z, save_path=None, title='2-DoF Pendulum Initialization'):
    _apply_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    # 1. 连杆
    ax.plot(x, z, 'o-', linewidth=3, color='#34495e', markersize=8, zorder=1, label='Links')

    # 2. 关节（排除基座和末端以示区分）
    if len(x) > 2:
        ax.scatter(x[1:-1], z[1:-1], s=100, c='#e74c3c', zorder=2, label='Joints')

    # 3. 基座
    ax.scatter(x[0], z[0], s=200, marker='^', c='black', zorder=3, label='Base (Fixed)')

    # 4. 末端执行器
    ax.scatter(x[-1], z[-1], s=150, c='#2ecc71', marker='*', zorder=3, label='End-Tip')

    # (Removed) zero reference line to reduce visual clutter; only the q=0.2 pose is shown

    # 注释与美化
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X Position [m]', fontsize=13)
    ax.set_ylabel('Z Position [m]', fontsize=13)
    ax.set_title(f"{title}\n($q_1 = \pi/2$ rad, $q_2 = 0$ rad)", fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.5))

    # 固定坐标范围为正方形，并将基座居中
    try:
        base_x = float(x[0]) if len(x) > 0 else 0.0
        base_z = float(z[0]) if len(z) > 0 else 0.0
        # 估计连杆长度用于确定可视半径
        diffs = np.sqrt(np.diff(x)**2 + np.diff(z)**2)
        link_len_est = float(np.mean(diffs)) if diffs.size > 0 else 1.0
        n_links_est = max(1, len(x) - 1)
        max_extent = n_links_est * link_len_est
        margin = 0.15 * max_extent
        # 计算基座到所有点的最大绝对偏移，确保基座在正方形中心
        dx = np.abs(x - base_x)
        dz = np.abs(z - base_z)
        half_span = max(np.max(dx) if dx.size>0 else 0.0, np.max(dz) if dz.size>0 else 0.0, max_extent) + margin
        # 设置为正方形范围，基座在中心
        ax.set_xlim(base_x - half_span/5.0, base_x + half_span)
        ax.set_ylim(base_z - half_span, base_z + half_span/5.0)
        ax.set_aspect('equal', adjustable='box')
    except Exception:
        pass

    # 累积弯曲注释（示例）
    # mid_idx = min(3, len(x)-1)
    # ax.text(x[mid_idx]+0.2, z[mid_idx], "Cumulative Curvature", fontsize=10, color='#34495e', style='italic')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        print('Saved:', save_path)
    plt.show()


def main():
    # 参数
    n_links = 2
    link_length = 1.0
    q = (1.57079632679, 0.0)

    x, z = compute_fk_2d(q, link_length=link_length)
    fig_path = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi_2p/fig/model/pendulum_2dof_2d.png')
    plot_chain(x, z, save_path=fig_path)
    print('x, z:', x, z)


if __name__ == '__main__':
    main()
