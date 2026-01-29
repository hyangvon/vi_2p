编译、仿真、画图
colcon build && ros2 run vi_2p vi_compare_sim.py  && ros2 run vi_2p rk4_vi_compare_sim.py

编译、画图（跳过仿真）
colcon build && ros2 run vi_2p vi_compare_sim.py --skip-sim && ros2 run vi_2p rk4_vi_compare_sim.py --skip-sim

编译、画热力图（对数）
colcon build && ros2 run vi_2p etsvi_sweep_heatmap.py --log