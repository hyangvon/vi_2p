#!/usr/bin/env python3
import os
import sys
import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image  # 需要 pip install pillow

# ---------------- 配置部分 ----------------
# 默认 URDF 文件名（会在多个候选位置搜索）
URDF_FILENAME = '2_pendulum.urdf'

def locate_urdf(filename):
    script_dir = os.path.dirname(__file__)
    checked = []
    # 常见候选位置：脚本包内的 urdf 目录
    checked.append(os.path.normpath(os.path.join(script_dir, '..', 'urdf', filename)))
    # 向上遍历祖先目录，尝试 workspace 根下的 src/vi_2p/urdf
    cur = script_dir
    while True:
        checked.append(os.path.normpath(os.path.join(cur, 'src', 'vi_2p', 'urdf', filename)))
        checked.append(os.path.normpath(os.path.join(cur, 'src', 'vi', 'urdf', filename)))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # 当前工作目录
    checked.append(os.path.normpath(os.path.join(os.getcwd(), filename)))

    for c in checked:
        if os.path.exists(c):
            print(f"找到 URDF: {c}")
            return c
    # 未找到则返回原始文件名（后续代码会报错并提示）
    print(f"未在候选位置找到 {filename}，将使用原始路径：{filename}")
    return filename

# 解析得到最终的 URDF 路径
URDF_PATH = locate_urdf(URDF_FILENAME)
OUTPUT_FILENAME = "figure_high_res.png"
IMG_WIDTH = 1920  # 想要多清晰都可以，推荐 1920 或 3840 (4K)
IMG_HEIGHT = 1080

# ---------------- 1. 初始化 (HEADLESS模式) ----------------
# 关键点：使用 p.DIRECT 而不是 p.GUI
if p.connect(p.DIRECT) < 0:
    print("无法连接到 PyBullet (DIRECT)。")
    sys.exit(1)

# 将 pybullet 自带数据路径 加入搜索路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# ---------------- 2. 加载环境与机器人 ----------------
# 加载地板（为了有阴影和参照，建议加载，不想要可以之后裁掉）
plane_id = p.loadURDF("plane.urdf")

# 检查 URDF 是否存在（优先使用相对/绝对路径）
if not os.path.isabs(URDF_PATH) and not os.path.exists(URDF_PATH):
    # 尝试在当前工作目录和脚本目录查找
    candidate = os.path.join(os.getcwd(), URDF_PATH)
    if os.path.exists(candidate):
        URDF_PATH = candidate
    else:
        script_dir = os.path.dirname(__file__)
        candidate2 = os.path.join(script_dir, URDF_PATH)
        if os.path.exists(candidate2):
            URDF_PATH = candidate2
        else:
            print(f"找不到 URDF 文件: {URDF_PATH}")
            print("请提供正确的 URDF_PATH（绝对路径或将文件放在当前目录/脚本目录）。")
            p.disconnect()
            sys.exit(1)

# 加载机器人（useFixedBase 可用 True/False）
robot_id = p.loadURDF(URDF_PATH, [0, 0, 1], useFixedBase=True)

# --- 为了在渲染中可视化基座，创建一个落地的长圆柱体（静态可视物体） ---
# 圆柱高度（m）和半径，可按需调整
cyl_height = 2.2
cyl_radius = 0.02
# 获取机器人基座位置，圆柱放置在地面上（底面 z=0），中心高度为 cyl_height/2
base_pos, _ = p.getBasePositionAndOrientation(robot_id)
cyl_center = [base_pos[0], base_pos[1], cyl_height / 2.0]
# 创建可视形状并生成一个无质量的静态 multi-body（无碰撞，仅用于渲染）
vis_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=cyl_radius, length=cyl_height, rgbaColor=[0.2, 0.2, 0.2, 1])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_id, basePosition=cyl_center)
# --- 圆柱体基座创建完毕 ---

# ---------------- 3. 设置关节角度 (图2/图5的状态) ----------------
# 例如图5的配置：所有关节 0.2 rad
num_joints = p.getNumJoints(robot_id)
# 设置期望角度（弧度）：关节1 = 90度, 关节2 = 0度
desired_angles = [np.pi/2, 0.0]
# 收集可动关节索引并按顺序赋值
movable_joints = []
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_type = info[2]
    if joint_type != p.JOINT_FIXED:
        movable_joints.append(i)

if len(movable_joints) == 0:
    print("警告：未找到可动关节。")
else:
    for idx, joint_idx in enumerate(movable_joints):
        angle = desired_angles[idx] if idx < len(desired_angles) else 0.0
        p.resetJointState(robot_id, joint_idx, angle)
        print(f"设置关节索引 {joint_idx} 为角度 {angle} rad")

# 可选：如果需要，可以在这里调整机器人基座位置，例如提高 base 的 z 值
p.resetBasePositionAndOrientation(robot_id, [0, 0, 2.2], [0, 0, 0, 1])

# 使用机器人基座位置作为相机的目标点，向上微调以避免被地面遮挡
base_pos, _ = p.getBasePositionAndOrientation(robot_id)
target_position = [base_pos[0], base_pos[1], base_pos[2] - 0.8]

# 可以调整这些值来改变取景（如果想看得更远/更整体，增大 distance；减少俯仰角）
camera_distance = 2.5
camera_yaw = 180
camera_pitch = -15

# ---------------- 4. 设置相机 (关键步骤) ----------------
# 我们需要计算 View Matrix (相机在哪里) 和 Projection Matrix (镜头畸变)

# 相机位置参数已在上方根据机器人基座计算并可被调整

view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=target_position,
    distance=camera_distance,
    yaw=camera_yaw,
    pitch=camera_pitch,
    roll=0,
    upAxisIndex=2
)

# 投影矩阵 (FOV等参数)
proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=float(IMG_WIDTH) / IMG_HEIGHT,
    nearVal=0.1,
    farVal=100.0
)

# ---------------- 5. 渲染并保存 ----------------
print("开始渲染...")
# 首先尝试硬件渲染（在有 GPU 和正确驱动的环境中可用），失败则回退到 TINY_RENDERER
try:
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        shadow=1,
        lightDirection=[1, 1, 1]
    )
except Exception:
    # 回退：使用软件渲染器
    print("硬件渲染不可用，回退到 TINY_RENDERER。")
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )

# 参考 plot_pendulum.py 的保存路径结构，保存到 package 下的 fig/model 目录
output_path = os.path.expanduser(os.path.join('~', 'ros2_ws', 'dynamic_ws', 'src', 'vi_2p', 'fig', 'model', OUTPUT_FILENAME))
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print(f"渲染完成，正在保存为 {output_path} ...")

# PyBullet 返回的是一个扁平的列表，需要转换成 numpy 数组
rgb_array = np.array(rgbImg, dtype=np.uint8)
# 有些环境下 rgbImg 已经是 (width*height*4,) 的扁平数组，也有可能直接是形状良好的数组
try:
    rgb_array = rgb_array.reshape((height, width, 4))  # RGBA
except Exception:
    # 尝试按 (width, height, 4) 重塑，再转置为 (height, width, 4)
    rgb_array = rgb_array.reshape((width, height, 4)).transpose((1, 0, 2))

rgb_array = rgb_array[:, :, :3]  # 去掉 Alpha 通道，保留 RGB

# 使用 PIL 保存图片
image = Image.fromarray(rgb_array)
image.save(output_path)

print("保存成功！请使用 scp 下载查看：", output_path)
p.disconnect()