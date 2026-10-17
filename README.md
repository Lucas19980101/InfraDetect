# InfraDetect 推理原型交付包

**只提交当前 `prototype_v1_delivery/` 这一个文件夹。**旁边的 `prototype_v1/`
是开发工作区，不属于交付内容；本交付文件夹内部也不再嵌套第二个 prototype。

这是供组员阅读和试运行的**仅推理（inference-only）**版本。GitHub 仓库包含统一命令行入口、
二维证据融合与三维核对代码、复核界面，以及 5 个模型的说明和配置。权重单独提供，
不上传 GitHub：取得 `models_weights.zip` 后，在仓库根目录解压，生成
`models/**/checkpoint.pt`，再执行下面的测试和推理命令。压缩包不应提交到仓库。
**不包含**数据集、训练脚本、训练数据加载器、旧实验输出或 COLMAP 重建文件。
YAML 中仍有 `train` 字段，用于保留模型设置和推理 AMP 开关；包内并无训练代码。
随包 checkpoint 保留模型权重，但移除了 optimizer、scheduler、scaler 状态，
不能直接用于续训。

本包是研究原型，**不是已验证的工程诊断产品**。输出的是模型证据和待复核候选，
不等于确诊空鼓、污渍或热桥，也不能代表新建筑上的系统准确率。

可选的墙面／空调证据仅影响 IR 候选的**人工复核顺序**：不会删除候选、修改原始
IR 概率，或生成经过定标的缺陷概率。当前权重尚未被目标建筑真值证明能提高准确率。

## 目录说明

- `src/`：统一命令行入口；`tests/`：交付包回归测试。
- `infradetect/`：场景融合、跨模态对应和三维复核界面。
- `infradetect/inference/`：热模型、RGB 模型和原分辨率滑窗推理。
- `infradetect/evidence/`：导出概率图、跨模态对应、融合及复核排序。
- `infradetect/geometry/`：点云投影、候选聚合、多视角核对和几何基线。
- `models/`：每个模型的配置与说明；单独解压权重包后，权重也位于对应子目录。
- `infradetect/examples/`：场景清单模板和示例配准矩阵。
- `ARCHITECTURE.md`：输入、处理和输出；`ENVIRONMENT.md`：安装环境。
- `requirements-inference.txt`：依赖配置；`pyproject.toml`：包入口元数据。

交付包将原先依赖训练模块的导入替换为 `infradetect/inference/` 中的仅推理模块；
原训练仓库未改动。模块如何协作、
结果应如何理解，见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 环境与快速检查

用 [requirements-inference.txt](requirements-inference.txt) 安装二维推理依赖；
PyTorch/torchvision 须按本机 CPU 或 CUDA 单独安装。只有运行 COLMAP 三维投影
及多视角核对时才额外需要 `pycolmap`。独立环境创建、安装顺序、已检查版本及
三维限制见 [ENVIRONMENT.md](ENVIRONMENT.md)。不要直接改动组员现有项目环境。

在交付包根目录执行：

```bash
export PYTHONPATH="$PWD/src"
python -m infradetect_prototype --help
python -m pytest -q -p no:cacheprovider tests
python -m infradetect_prototype smoke --output /path/to/new_smoke_directory
```

`smoke` 不需要外部数据集：它生成一张很小的**合成**四通道图，加载随包五个权重，
走完二维九阶段并在新目录保存结果。`complete=true` 只证明单文件夹的基本
功能闭环，**不能**用合成图判断模型准确率。输出目录必须是新的或空的。

5 个权重由单独的压缩包提供；功能、指标和适用边界见 [models/README.md](models/README.md)。
RGB 六类权重现为 Experiment B；其初始化、原尺度采样、验证协议和 C70
对照结果写在[六类模型卡](models/rgb/visible_defects/README.md)。这些是训练
**记录说明**，不表示本包包含训练代码或数据。
标准输入是一张 `H×W×4` NumPy 图像，通道顺序为
`[B,G,R,Thermal]`；旧 TBBRv2 五通道文件仍可读，第五通道不参与当前推理。
把[场景清单示例](infradetect/examples/scene.example.json)
复制为新的 manifest，填写真实图像路径、场景 ID、来源和配准信息。示例中的单位变换
**只适用于已同像素对齐的场景栈**；其他 RGB/IR 图像对必须先核验配准。
字段级说明见 [场景清单说明](infradetect/examples/README.md)；模型 YAML 的
用途与限制见 [模型配置说明](models/CONFIGS.md)。
先做只读预检，再运行：

```bash
python -m infradetect_prototype check --manifest /path/to/your_scene.json
python -m infradetect_prototype run \
  --manifest /path/to/your_scene.json --output /path/to/new_output_directory
python -m infradetect_prototype summary --output /path/to/new_output_directory
```

`check` 只检查图像、权重和清单是否具备运行条件，不加载模型或执行推理。
`run` 拒绝覆盖非空输出目录；每次运行须选择新目录。输入数据由使用者在分享本包后
单独提供，包内不附带真实示例数据；上述 `smoke` 输入运行时才生成。

## 点云查看与三维阶段

点云／图片复核界面在 `infradetect/tbbr_review.py` 与 `tbbr_review.html`。
运行 `python -m infradetect.tbbr_review --help` 查看所需的 COLMAP 和复核输入。
统一 CLI 另提供 `project-3d`、`aggregate-3d`、`compare-views`；各命令可用
`--help` 查看参数。多视角图像和点云必须来自**同一次重建、同一坐标系**。
成功运行单张图的二维推理，不代表已完成三维重建或多视角核对。

已有 COLMAP 重建时，先用统一的 geometry 接口检查文件是否齐备：

```bash
infradetect-prototype geometry-check \
  --model /path/to/original/sparse \
  --workspace /path/to/dense \
  --ply /path/to/dense/fused.ply
```

`ready_for_projection=true` 只表示原始相机模型、稠密工作区相机模型、
几何深度图和点云文件存在；它**不能**证明这些文件来自同一次重建。
后续 `project-3d` 还需要二维融合结果和 RGB→IR 配准文件。
目前交付包的几何后端是现有 COLMAP 重建，不会自行重建三维，也未附带新模型权重。

### 几何异常基线（暂用）

工作区原有的 `Real3D-AD/building-anomaly` 做过墙面平面偏离实验。
交付包提供一个独立的 `geometry-baseline` 入口，复用其核心思路：
输入**裁出的单面墙点云**，拟合主平面，输出原 PLY 点顺序对应的
`point_scores.npz`（`scores` 为绝对距离，`candidate` 为候选布尔值）
和 `report.json`。运行前安装可选依赖 `open3d`：

```bash
python -m pip install 'open3d>=0.19,<0.20'
infradetect-prototype geometry-baseline \
  --ply /path/to/single_wall.ply \
  --output /path/to/new_geometry_result
```

默认拟合阈值和候选阈值分别为点云包围盒对角线的 0.002、0.006；
这不是米制阈值。不要直接把整栋楼的 `fused.ply` 当作单面墙输入：
另一面墙、窗、管道、栏杆和重建噪声都可能被标为候选。
基线不需要训练权重，也**未经 BFDD/DamCrack 真实几何缺陷验证**；
输出只供复核与后续替换模型 A/B，不等于确认病害。

要在已有的三维复核网页查看这些候选，运行：

```bash
python -m infradetect.tbbr_review \
  --geometry-result /path/to/new_geometry_result \
  --geometry-ply /path/to/single_wall.ply \
  --review-dir /path/to/new_geometry_reviews \
  --port 8766
```

浏览器打开 `http://127.0.0.1:8766/`。橙点是几何偏离候选；
左侧按分数列出，点击可在三维点云中定位。可给展示的点记录
“值得复核／正常构件／重建噪声／不确定”，结果另存为
`--review-dir/reviews.json`，不改原始点云或候选分数。
为保证浏览器响应，超过 5,000 个候选时只展示分数最高的 5,000 个，
总候选数仍会显示。只传基线结果时，geometry 模式没有照片；
下面的相机和深度参数可打开经过遮挡核验的照片对应。

如果单面墙 PLY 与某次 COLMAP 重建处于**同一坐标系**，可补上相机和
稠密深度图，让三维候选链接到真正可见的 RGB 照片：
此模式需在**同一个 Python 环境**安装 `open3d` 与 `pycolmap`；
二维推理环境默认未包含这两个可选依赖。

```bash
python -m infradetect.tbbr_review \
  --geometry-result /path/to/new_geometry_result \
  --geometry-ply /path/to/single_wall.ply \
  --geometry-model /path/to/original/sparse \
  --geometry-workspace /path/to/dense \
  --images /path/to/original/rgb_images \
  --review-dir /path/to/new_geometry_reviews \
  --port 8766
```

照片位置使用原始相机投影，并用稠密深度图排除被遮挡的点。
如另有配对照片清单和已核验的 RGB→IR 单应矩阵，可再加
`--source-map /path/to/source_map.json` 与
`--geometry-homographies /path/to/homographies`，才会显示 IR 图上的定位圈。
没有配准矩阵时不在 IR 上猜位置；没有相机/深度图时仍可只看点云。

界面会把**展示出来**且空间邻近的候选点组成组；距离上限受点云尺度
约束，避免跨墙合并。组级判断和备注保存在
`--review-dir/group_reviews.json`，逐点判断仍保存在 `reviews.json`。
组只方便人工浏览，并不表示组内点已被确认属于同一缺陷。
复核目录会记录所用几何结果的身份；换点云或换基线结果时请使用新的
`--review-dir`，避免相同点编号误读成旧判断。

## 来源、授权与安全

本包从 InfraDetect 研究工作区于 2026-10-15 选取源码制作，后续仅同步了推理证据
排序与说明。分发时应保留上游许可并核查数据集使用条款；向组外分享**含模型权重**
的版本前，还须确认 checkpoint 的再分发权限。不要在 manifest 中填写隐私数据或
环境密钥，因为运行会保存 manifest 快照用于复现。
