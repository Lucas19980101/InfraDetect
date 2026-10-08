# FAST-LIVO2 论文与代码导读

本文对应论文 **FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry**
（arXiv:2408.14035）。目标是先建立整体心智模型，再进入函数细节。论文描述算法，代码还包含
ROS 数据接入、不同 LiDAR 适配、可视化和文件保存；二者不是逐文件一一对应。

## 先理解一句话

FAST-LIVO2 用 IMU 预测运动状态，以相机时间重新切分 LiDAR 点流；在同一个 19 维误差状态
迭代卡尔曼滤波器（ESIKF）里，先用 LiDAR 点到平面残差更新，再用图像 patch 的光度残差更新。
LiDAR 建立带平面统计量的自适应 voxel/octree 地图，视觉模块在其中一些 3D 点上附加图像 patch。

```text
LiDAR callback ─→ preprocess ─┐
IMU callback ────────────────┼→ sync_packages（按相机时刻重组 scan）
RGB callback ────────────────┘                 │
                                               ▼
                         IMU propagation + LiDAR 去畸变
                                               │ 先验 x̂, P̂
                                               ▼
                   LIO：point-to-plane ESIKF update
                                               │ 中间后验
                                               ▼
                   VIO：sparse-direct photometric ESIKF update
                                               │ 最终后验
                     ┌─────────────────────────┴──────────────────┐
                     ▼                                            ▼
             更新统一 voxel map                         发布 pose / 彩色点云
```

论文称它为“sequential update”：LiDAR 与相机不是各维护一套状态后再融合，而是依次更新同一个
状态。代码用 `LidarMeasures.lio_vio_flg` 在相邻两次主循环中执行 LIO、VIO。

## 从哪里开始读

建议按以下顺序阅读：

1. `src/main.cpp`：ROS 入口，只有创建和启动 `LIVMapper`。
2. `include/common_lib.h`：状态、测量包、坐标点及枚举，是全项目的数据字典。
3. `src/LIVMapper.cpp`：完整控制流，先读 `run()`、`sync_packages()`、`handleLIO()`、`handleVIO()`。
4. `src/IMU_Processing.cpp`：读 `Process2()`，再追到 `IMU_init()` 和 `UndistortPcl()`。
5. `src/voxel_map.cpp`：读 `StateEstimation()`、`BuildResidualListOMP()`、`build_single_residual()`。
6. `src/vio.cpp`：先读 `processFrame()`，再按它调用的七个阶段向上阅读。
7. `include/frame.h`、`include/feature.h`、`include/visual_point.h`：理解视觉数据之间的关系。

第一次阅读可跳过 ROS publisher、调试计时、RViz marker 和各 LiDAR 厂商 handler 的细节。

## 文件与模块

| 文件 | 作用 | 论文对应 |
| --- | --- | --- |
| `src/main.cpp` | 初始化 ROS、构造并运行总控制器 | 工程入口 |
| `include/common_lib.h` | 19 维状态、测量包、坐标点结构和模式枚举 | Sec. IV-A |
| `include/LIVMapper.h`, `src/LIVMapper.cpp` | 读参数、订阅、时间同步、LIO/VIO 状态机、发布和保存 | Fig. 2、Algorithm 1 |
| `include/preprocess.h`, `src/preprocess.cpp` | 将 Avia/Ouster/Velodyne/Hesai/Robosense 等消息统一为 `PointCloudXYZI`，保存点内时间并滤点 | LiDAR 输入前端 |
| `include/IMU_Processing.h`, `src/IMU_Processing.cpp` | IMU 初始化、状态与协方差传播、LiDAR 点运动补偿 | Sec. IV-B/C，Fig. 3 |
| `include/voxel_map.h`, `src/voxel_map.cpp` | 哈希根 voxel、自适应 octree、平面拟合与更新、点到平面 ESIKF | Sec. V-A/B、Sec. VI |
| `include/vio.h`, `src/vio.cpp` | 可见视觉点选择、raycasting、patch warp、直接光度更新、法线和参考 patch 更新 | Sec. V-C/D/E、Sec. VII |
| `include/frame.h`, `src/frame.cpp` | 一帧灰度图、金字塔、相机模型及 world-to-camera 位姿 | Sec. VII 坐标/投影 |
| `include/feature.h` | 某个 3D 点在一帧图像中的 patch 观测 | Sec. V-C/D |
| `include/visual_point.h`, `src/visual_point.cpp` | 世界系 3D 位置、法线、协方差和多帧 patch 观测 | Sec. V-C/D/E |
| `launch/*.launch`, `config/*.yaml` | 数据集入口、topic、外参、噪声和算法开关 | Sec. VIII/IX 参数设置 |

`CMakeLists.txt` 中的库名也体现了划分：`pre`、`imu_proc`、`lio`、`vio`、`laser_mapping`，
最后链接成 `fastlivo_mapping`。项目以 CPU、Eigen/PCL/OpenCV/OpenMP 运算为主，没有 CUDA kernel。

## 核心数据结构

### `StatesGroup`：滤波器状态

代码中的状态排列是：

```text
rotation(3), position(3), inverse exposure(1), velocity(3),
gyro bias(3), accel bias(3), gravity(3) = 19
```

它对应论文式 (2)。旋转属于 SO(3)，所以 `operator+` 不是逐元素相加，而是
`R_new = R * Exp(delta_theta)`；`operator-` 用 `Log(R_old^T R_new)` 得到误差旋转。这就是论文
中的 boxplus/boxminus 在代码里的实现。`cov` 是该误差状态的 19×19 协方差。

### `pointWithVar`：带不确定性的 LiDAR 点

一个物理点同时保存 LiDAR body、IMU body 和 world 三个坐标，以及测量协方差、状态传播后的
协方差和关联平面法线。LIO 用这些量计算点到平面的残差权重；VIO 用世界点和法线生成视觉点。

### `VoxelOctoTree` 与 `VoxelPlane`

世界坐标按固定根 voxel 哈希索引。每个根 voxel 内部是自适应 octree：如果点集可拟合平面，
叶节点保存中心、法线、协方差；否则继续八分直到最大层。它避免全局最近邻树搜索，并为视觉
patch warp 直接提供局部平面先验。

### `Frame`、`VisualPoint`、`Feature`

```text
Frame
  T_f_w_：world → 当前 camera
  img_：当前灰度图

VisualPoint
  pos_：世界坐标 3D 点
  normal_：世界坐标平面法线
  obs_ ───────────────┐
                      ▼
Feature
  px_：该帧像素
  patch_：像素邻域灰度
  T_f_w_：产生这次观测时的相机位姿
  point_：回指 VisualPoint
```

`Feature` 在这里不是 ORB/SIFT 描述子，而是一小块原始灰度 patch。FAST-LIVO2 是直接法：优化
patch 像素亮度差，不先做关键点描述子匹配。

## 一帧数据实际怎样运行

### 1. 回调与预处理

`initializeSubscribersAndPublishers()` 创建三个订阅。LiDAR 回调调用 `Preprocess::process()`：

- 删除盲区、非法点并按 `point_filter_num` 降采样；
- 将厂商字段转换为 PCL 点；
- 将每点相对 scan 起点的时间换成毫秒，暂存在 `PointType::curvature`。

这里借用 `curvature` 只是为了复用 PCL 点类型，并不表示真正的曲率。

### 2. Scan recombination

论文 Sec. IV-B 不是简单挑时间最近的 LiDAR 帧。`sync_packages()` 取
`image_stamp + img_time_offset + exposure_time_init`，然后逐点比较
`lidar_header_stamp + curvature/1000`，把相机时刻之前和之后的点分到
`pcl_proc_cur`、`pcl_proc_next`。这样 LIO 与 VIO 可在同一个相机时刻更新。

### 3. IMU 传播和去畸变

`ImuProcess::Process2()` 在启动时先由多帧 IMU 均值估计重力与零偏。正常阶段中，
`UndistortPcl()`：

1. 按 IMU 时间正向积分姿态、位置、速度并传播协方差；
2. 保存 scan 中间的 IMU pose；
3. 从 scan 尾部反向遍历 LiDAR 点，将不同采样时刻的点补偿到更新时刻。

输出是预测状态和去畸变点云，是后续两个测量更新的共同先验。

### 4. LiDAR 顺序更新

`handleLIO()` 对点云体素降采样、估计每点测量协方差并变换到世界系，然后调用：

```text
VoxelMapManager::StateEstimation
  └─ BuildResidualListOMP
       └─ build_single_residual
```

`build_single_residual()` 在点所在 voxel/octree 中找平面，残差本质是：

```text
r = nᵀ (p_world - q)
```

其中 `n`、`q` 是地图平面的法线和中心。`StateEstimation()` 计算残差对旋转、平移的雅可比，
结合点和平面的不确定性形成权重，执行论文式 (11) 的迭代更新。收敛后更新状态协方差，再由
`UpdateVoxelMap()` 把新 scan 融入统一地图。

### 5. Visual map point selection

`VIOManager::retrieveFromVisualSparseMap()` 对应论文 Sec. VII-A：

1. 从本帧 LiDAR 点命中的 voxel 和上一帧可见 voxel 中取视觉点；
2. 若 `raycast_en` 开启，对仍为空的图像网格沿视线按深度采样 voxel；
3. 每个图像网格保留合适点；
4. 以负深度、出视场、深度不连续、视角和光度误差剔除外点。

Raycasting 只在可见 voxel 查询不足时按需发生，不构建一张完整稠密射线图，因此论文称
“on-demand”。当前 `avia.yaml` 的 `raycast_en: false`，要复现实验中的该模块需显式开启。

### 6. 平面感知的 affine patch warping

参考帧和当前帧视角不同，直接比较同一方形像素窗口会产生形变。
`getWarpMatrixAffineHomography()` 用 3D 点的深度、平面法线和相对相机位姿计算局部单应变换，
`warpAffine()` 将参考 patch 变换到当前视角。论文 Fig. 5/S1 展示了平面先验比恒深假设更准确。

### 7. 稀疏直接视觉 ESIKF

`computeJacobianAndUpdateEKF()` 从粗到细处理图像金字塔。每个像素残差可概括为：

```text
r = tau_ref * I_ref(warped_pixel) - tau_cur * I_cur(pixel)
```

其中 `tau` 是状态中的相对逆曝光时间。`updateState()` 或 `updateStateInverse()` 计算图像梯度、
投影雅可比和位姿链式雅可比，然后更新同一个 `StatesGroup`。这对应论文 Sec. VII-B 与
Algorithm 1 后半段。

### 8. 视觉地图维护

- `generateVisualMapPoints()`：在有 LiDAR 平面支撑、图像梯度足够的网格建立新视觉点；
- `updateVisualMapPoints()`：运动基线或像素视差足够时保存新 patch 观测；
- `updateReferencePatch()`：从多帧观测选择清晰、正视且光度一致的参考 patch，并细化法线；
- 法线收敛后删除非参考 patch，控制内存。

这对应论文 Sec. V-C/D/E。补充材料 S2/S3 分别展示 affine warp、参考 patch 更新和法线收敛
对地图清晰度的影响。

### 9. 输出

`publish_odometry()` 输出 IMU/body 在 `camera_init` 世界系中的 pose；
`publish_frame_world()` 把世界点投影进当前 RGB 图并采色，发布 `/cloud_registered`；退出时
`savePCD()` 写出 `Log/pcd/all_raw_points.pcd` 与降采样彩色 PCD。内存中的 octree、plane 和
patch 默认不是 PCD 的一部分。更详细的 topic、坐标系与新增 RGB–3D 表见
`DATA_FLOW.zh-CN.md`。

## 配置怎样影响论文模块

当前 `config/avia.yaml` 中值得先看的参数：

| 参数 | 含义 |
| --- | --- |
| `extrin_calib.*` | LiDAR→IMU、IMU/LiDAR→camera 的刚体外参 |
| `time_offset.*` | 三传感器固定时间偏移及初始曝光时刻偏移 |
| `preprocess.lidar_type` | 决定 ROS 消息和厂商 handler |
| `lio.voxel_size/max_layer` | 根 voxel 与内部 octree 深度 |
| `vio.patch_size/patch_pyrimid_level` | 光度残差 patch 大小与金字塔层数 |
| `vio.normal_en` | 使用 LiDAR 平面先验并执行法线细化 |
| `vio.raycast_en` | 开启论文的按需 raycasting |
| `vio.inverse_composition_en` | 在两套视觉雅可比实现之间切换 |
| `vio.exposure_estimate_en` | 把相对逆曝光作为状态共同估计 |
| `pcd_save.*` | 保存世界/机体系点云、间隔与降采样尺寸 |

## 代码与论文之间的几个易混点

1. 论文说“single unified voxel map”，并不表示代码只有一个容器。几何 plane map 是统一的
   3D 基础；VIO 另有 `feat_map` 索引 `VisualPoint`，视觉点从几何 map 产生并使用其平面先验。
2. `Frame::T_f_w_` 是 world-to-camera，而 `_state.rot_end/pos_end` 表示 body-to-world；方向相反。
3. `curvature` 在预处理后的点里代表毫秒时间偏移，不是曲率。
4. `normal_en` 不只是显示法线，它决定是否使用平面单应 warp 和法线 refinement 路径。
5. `dense_map_en` 控制发布策略；视觉估计本身仍是 sparse-direct，不是对每个 RGB 像素优化。
6. 论文应用部分的 VDBFusion/OpenMVS/3DGS 是 FAST-LIVO2 输出之后的独立流水线，不在本仓库
   核心定位代码中。仓库的 `scripts/reconstruct-mesh.sh` 是后来加入的本地表面重建工具。

## 阅读时设置断点

用 Retail Street bag 调试时，推荐按控制流设置这些断点：

```text
LIVMapper::sync_packages
ImuProcess::Process2
ImuProcess::UndistortPcl
LIVMapper::handleLIO
VoxelMapManager::StateEstimation
VoxelMapManager::build_single_residual
LIVMapper::handleVIO
VIOManager::retrieveFromVisualSparseMap
VIOManager::updateState
VIOManager::generateVisualMapPoints
VIOManager::updateReferencePatch
```

每到一个断点先观察 `_state`、`LidarMeasures.lio_vio_flg`、当前点数、`voxel_map_.size()`、
`visual_submap->voxel_points.size()`，比直接单步进入 Eigen/PCL/OpenCV 模板代码更容易理解。

## 论文位置索引

- Fig. 2：整体系统结构；
- Sec. IV 与 Algorithm 1：状态、scan recombination、传播和顺序 ESIKF；
- Sec. V：统一自适应 voxel map、视觉点、参考 patch 与法线更新；
- Sec. VI：LiDAR 点到平面测量模型和不确定性；
- Sec. VII：可见点/raycasting、外点剔除和稀疏直接光度模型；
- Sec. X-C：用彩色点云与相机 pose 接 VDBFusion、OpenMVS、3DGS；这些属于下游应用。

论文链接：<https://arxiv.org/abs/2408.14035>；仓库还附有
`Supplementary/LIVO2_supplementary.pdf`，其中包含各模块消融实验和更多可视化。
