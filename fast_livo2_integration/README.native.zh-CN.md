# 在当前 WSL 中直接运行

此方案在 Ubuntu 26.04 / WSL2 内通过 RoboStack + Conda 安装 ROS Noetic 和 C++ 库，不使用 Docker，也不要求 CUDA。环境位于 `.native/env`，源码依赖、编译产物和 ROS 日志均位于已被 Git 忽略的 `.native/`。

## 安装与编译

在仓库根目录执行，需要现有 Conda 和联网：

```bash
conda env create --prefix "$PWD/.native/env" --file environment-native.yml --solver libmamba -y
bash scripts/native-sources.sh
bash scripts/native-build.sh
```

环境已经安装时不要重复创建，直接执行构建脚本。默认用 2 个并行编译任务，可通过 `BUILD_JOBS` 调整。源码计算线程数仍由项目 CMake 决定，默认最多 4 个，与编译并行度不同。

`native-sources.sh` 固定四个源码仓库的 commit；`native-patch-deps.py` 对这些版本应用兼容补丁。Conda 安装版本可以导出，便于同平台复现：

```bash
conda list --prefix .native/env --explicit > .native/conda-explicit.txt
```

## 启动和回放

每个新终端先进入仓库根目录，然后激活：

```bash
source scripts/native-env.sh
roslaunch fast_livo mapping_avia.launch rviz:=false
```

回放数据时，先启动 `roscore`，在启动 mapping 节点之前执行 `rosparam set use_sim_time true`；另一个已激活的终端执行：

```bash
rosbag play --clock /absolute/path/to/data.bag
```

默认参数对应作者的 Avia 数据。自己的设备需要修改 `config/avia.yaml` 中的话题、雷达类型、外参、时间偏移，以及 `config/camera_pinhole.yaml` 的相机参数。

图形界面可用后，可去掉 `rviz:=false` 开启 RViz；RViz 的 OpenGL 显示不代表定位算法使用 GPU。离线网格脚本 `scripts/mesh.py` 的 Python 依赖不包含在本环境中。

输入 topic、时间同步、位姿/地图坐标系、RGB–3D 对应导出及异常检测数据桥接见
[`DATA_FLOW.zh-CN.md`](DATA_FLOW.zh-CN.md)。

结合论文理解入口、状态、IMU、LIO、VIO、统一体素地图与关键函数，见
[`CODE_WALKTHROUGH.zh-CN.md`](CODE_WALKTHROUGH.zh-CN.md)。

## 点云表面重建

FAST-LIVO2 保存的是 PCD 点云。对街景点云默认执行局部 GP3 三角化：

```bash
bash scripts/reconstruct-mesh.sh Log/pcd/all_downsampled_points.pcd retail_street
pcl_viewer Log/mesh/retail_street_gp3.vtk
```

脚本先估计法线，再生成 VTK 和 PLY 三角网格。`pcl_viewer` 查看 VTK；Blender、MeshLab 或 CloudCompare 可打开 PLY。可用 `NORMAL_K=30 GP3_RADIUS=0.25` 调整局部连接范围。Poisson 仅适合较完整的封闭物体，可将第三个参数设为 `poisson` 并用 `POISSON_DEPTH=9` 调整细节。

本机验证结果：ROS Noetic 1.17、OpenCV 4.10、PCL 1.14、Eigen 3.4 和 GCC 15 环境安装成功；主程序完成 Release 编译和链接。`roslaunch fast_livo mapping_avia.launch rviz:=false` 能启动 master、`laserMapping` 和图像转发节点，成功读取示例相机参数，并在没有传感器消息时保持等待。

## 参考

- [RoboStack 安装文档](https://robostack.github.io/GettingStarted.html)
- [RoboStack Noetic 包列表](https://robostack.github.io/noetic.html)
