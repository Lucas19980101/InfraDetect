# FAST-LIVO2 数据流、坐标系与 RGB–3D 对应

本文针对当前 `mapping_avia.launch` 和 `config/avia.yaml`。Retail Street 数据包位于
`/home/lucas19980101/projects/datasets/FAST-LIVO2/Retail_Street.bag`，包含 1355 帧图像、
1355 帧 Livox 点云和 27447 条 IMU 消息。

## 1. 输入和时间同步

| 传感器 | 默认 topic | ROS 数据类型 | 使用的时间 |
| --- | --- | --- | --- |
| RGB | `/left_camera/image` | `sensor_msgs/Image`，进入算法时转为 BGR8 | `header.stamp + img_time_offset + exposure_time_init` |
| LiDAR (Avia) | `/livox/lidar` | `livox_ros_driver/CustomMsg` | 帧头 `header.stamp`；点内 `offset_time` 被换算为毫秒并暂存在 PCL `curvature` 字段 |
| IMU | `/livox/imu` | `sensor_msgs/Imu` | `header.stamp - imu_time_offset` |

非 Livox 雷达走 `sensor_msgs/PointCloud2` 回调，并在帧头时间上增加
`lidar_time_offset`。当前 Avia 回调没有应用这个参数，这是原代码两条输入路径的差异。

LIVO 模式以校正后的图像曝光时刻为切分点：等待 LiDAR 和 IMU 都覆盖该时刻，将点内时间
早于切分点的 LiDAR 点放入本次 LIO，将 IMU 消息取到同一时刻；随后对同一时刻的图像做
VIO 更新。它依赖三个传感器使用同一时钟域，配置中的 offset 只补偿固定偏移。

对应实现：`initializeSubscribersAndPublishers()`、`livox_pcl_cbk()`、`imu_cbk()`、
`img_cbk()` 和 `sync_packages()` 均在 `src/LIVMapper.cpp`；Livox 点内时间换算在
`src/preprocess.cpp`。

## 2. 位姿和坐标系

状态量 `_state.rot_end`、`_state.pos_end` 表示 IMU/body 坐标到初始化世界坐标的位姿，
世界 frame 名为 `camera_init`。以下输出都使用这个状态：

- `/aft_mapped_to_init`：`nav_msgs/Odometry`，父 frame `camera_init`，子 frame `aft_mapped`。
- `/path`：`nav_msgs/Path`，其中 pose 在 `camera_init` 中。
- `/mavros/vision_pose/pose`：`geometry_msgs/PoseStamped`，frame 为 `camera_init`。
- `Log/pcd/lidar_poses.txt`：启用 PCD 保存后写出的逐帧 LiDAR/body 轨迹。

视觉模块的 `Frame::T_f_w_` 方向相反，它是 world-to-camera：

```text
p_camera = R_cw * p_world + t_cw
pixel = camera_model.project(p_camera)
```

新增的 `frames.csv` 保存的就是这个 `T_cw`，不是相机在世界中的 `T_wc`。若需要相机中心，
使用 `C_world = -R_cw^T t_cw`。

## 3. 地图输出

运行时有两套地图：

- LIO 几何地图是 `unordered_map<VOXEL_LOCATION, VoxelOctoTree*>`，保存体素、平面统计量和
  分层八叉树。
- VIO 稀疏地图是 `unordered_map<VOXEL_LOCATION, VOXEL_POINTS*>`，每个 `VisualPoint`
  保存世界坐标、法线、协方差以及有限数量的 patch/观测引用。

原项目不会序列化这两套内存结构。退出时保存的是累计注册点：当前配置
`pcd_save.type: 0` 和 `img_en: 1` 生成世界坐标下的彩色 `PointXYZRGB`：

- `Log/pcd/all_raw_points.pcd`：累计原始彩色点；
- `Log/pcd/all_downsampled_points.pcd`：以 `filter_size_pcd` 体素降采样后的彩色点；
- `Log/pcd/lidar_poses.txt`：轨迹文本。

PCD 只有 `x y z rgb`，不包含 voxel/octree、视觉 patch 或 frame 可见关系。

## 4. RGB 与 3D 的对应

主投影入口是 `Frame::w2c()`（`include/frame.h`）：先用 `T_f_w_` 把世界点变到相机系，
再调用相机模型投影。`LIVMapper::publish_frame_world()` 在正深度且像素落入图像时，通过
`VIOManager::getInterpolatedPixel()` 双线性采样 BGR，为 LiDAR 点着色。

反向从单个 RGB 像素不能唯一恢复 3D，因为它只定义一条相机射线。现在可用保存的观测表
把一个图像区域关联到当帧可见的 VIO 稀疏点；该表只覆盖 VIO 稀疏子图，不覆盖最终 PCD
中的每一个稠密 LiDAR 点，也没有完整 z-buffer 遮挡判断。

原代码的 `VisualPoint::obs_` 会限制为最多 30 条，点收敛后还会删除非参考 patch，并且退出
时完全不保存。因此新增了运行期持久化导出。每次运行创建：

```text
Log/correspondence/run_<启动时间>/
├── frames.csv        # frame ID、传感器时间、图像路径、尺寸、T_cw
├── points.csv        # 稳定 point ID、世界坐标、世界法线
├── observations.csv  # frame ID、point ID、u、v、相机深度
└── images/<frame>.png
```

`Log/correspondence/latest_run.txt` 指向最近一次运行。查询示例：

```bash
# 一个 3D 点在哪些帧/像素中出现
python scripts/query-correspondence.py point 123

# 图像 frame 20 中矩形 [100,200]–[300,400] 对应哪些稀疏 3D 点
python scripts/query-correspondence.py region 20 100 200 300 400
```

通过 `config/avia.yaml` 的 `correspondence.save_en` 和 `save_images` 控制导出。保存 PNG 会增加
磁盘写入和容量占用；只需要几何关系时可关闭 `save_images`。

## 5. Retail Street 到 3D anomaly pipeline

已从 `Log/pcd/all_downsampled_points.pcd` 自动选择最密集的 10 m 立方场景块，生成：

```text
Log/anomaly_bridge/incoming/retail_street_block.npz
```

它含 11247 个有限 `float32` 点、同顺序的全零 `uint8 mask`、原 PCD 点索引和裁剪参数，
符合 `../Real3D-AD/synthetic_wall_dataset.py` 的 NPZ 数据契约。可重复或指定建筑坐标：

```bash
source scripts/native-env.sh
python scripts/export-anomaly-patch.py \
  Log/pcd/all_downsampled_points.pcd \
  Log/anomaly_bridge/incoming/retail_street_block.npz \
  --size 10

# 手工选定建筑中心（世界坐标，单位米）
python scripts/export-anomaly-patch.py INPUT.pcd OUTPUT.npz \
  --center X Y Z --size 10
```

当前块是待检测输入，不能同时当训练基准并产生可信异常结论。运行 PatchCore 前还需要同一区域、
同一采集条件下的正常参考扫描，放入 `dataset/train/normal/`；测试文件放入
`dataset/test/normal/`，或在有逐点真值 mask 时放入 `test/bulge/`、`test/dent/`。然后在
Real3D-AD 的 `.venv` 中运行 `main_synthetic_wall.py --backend fpfh`。PointMAE 路径按现有实现
需要 CUDA，FPFH 路径可用 CPU。
