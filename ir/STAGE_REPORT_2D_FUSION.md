# InfraDetect 二维分割与 RGB–IR 融合阶段报告

**报告日期：** 2026-09-29  
**阶段状态：** 二维训练、RGB–IR 配准、Dataset_1x 全量融合及 COLMAP 局部三维投影验证已完成；FAST-LIVO2 全局坐标接入尚未开展。

## 1. 项目目标与本阶段范围

InfraDetect 的最终目标是把建筑缺陷定位到三维模型。本阶段完成以下二维处理链：

1. RGB 建筑构件分割，用于区分可检测墙面、门窗开口及其他区域；
2. IR 缺陷分割，用于预测 BFDD 定义的五类缺陷；
3. RGB 附属设备识别，用于标记空调等热干扰区域，而不是直接删除这些区域；
4. 使用 MFDN 估计 RGB→IR 单应变换；
5. 融合构件、缺陷、设备影响和边界不确定性，生成可供三维投影的状态图。

当前方案不要求逐像素绝对温度。热成像伪彩中的红色仅代表当前色标下相对偏热，不能单独作为缺陷判据。

## 2. 数据与标签

### 2.1 Dataset_1x / BFDD

实际数据共包含 838 对同名样本：

- `RGB/`：838 张 JPG；
- `IR/`：838 张 PNG；
- `Label/`：838 张单通道像素标签 PNG。

IR 标签映射为：

| ID | 类别 | 颜色 |
|---:|---|---|
| 0 | Background | `(0, 0, 0)` |
| 1 | Cracks | `(61, 61, 245)` |
| 2 | Peeling | `(169, 36, 191)` |
| 3 | Hollow Areas | `(174, 79, 13)` |
| 4 | Stains | `(36, 179, 83)` |
| 5 | Erosion | `(203, 253, 0)` |

BFDD 按图像分组划分，避免相邻或同组图像跨集合泄漏：

| 划分 | 数量 |
|---|---:|
| Train | 581 |
| Validation | 105 |
| Test | 152 |
| 合计 | 838 |

### 2.2 RGB 构件数据

构件任务最终合并了 3DOM、CMP Base、CMP Extended、eTRIMS、HZNU 和人工修正的 BFDD RGB 标签。统一为三个工程类别：

- `building_surface`：墙面及应保留为建筑表面的构件；
- `opening`：窗、门及相关开口；
- `other`：天空、遮挡物、非建筑背景及其他不应作为墙体缺陷载体的区域。

CMP 中 cornice、molding、pillar、deco 等保守并入建筑表面；sill、balcony、shop blind 等按与门窗开口的关系归入开口。天空和一般障碍物不再各自设类，统一并入 `other`。

最终适配清单共 1,583 条：

| 划分 | 数量 |
|---|---:|
| Train | 1,035 |
| Validation | 205 |
| Test | 203 |
| Exclude | 140 |

其中 BFDD RGB 人工适配样本为 66 张：48 train、10 validation、8 test。HZNU 按建筑编号成组划分，同编号的 original 与 perspective correction 不跨集合；经人工复核的 140 张不可靠样本进入 `exclude`。

### 2.3 空调与设备标签

使用 LabelMe 对 Dataset_1x 中的空调及相关设备区域进行增量标注。补丁训练清单共 80 个样本：

- 完整图：38；
- 空调正样本裁块：31；
- 困难负样本裁块：11；
- 划分：60 train、16 validation、4 test。

空调分支在融合阶段承担“设备影响提示”，不把相关热异常从最终结果中永久删除。

为检查跨数据集泛化，另从 HKUST `Facade_RGB_IR` 官方 train/validation 划分中人工复核 40 张对齐图像，其中 32 张用于训练、8 张用于验证；标注包含 `opening`、`other` 和独立的 `air_conditioner` 多边形。HKUST 官方 test 划分未参与训练或阈值选择。该增量集包含 7 张空调正样本，其余无空调图像作为窗框、百叶和墙面困难负样本。

## 3. 模型与训练方案

### 3.1 RGB 构件分割

- 模型：SegFormer-B0；
- 类别数：3；
- 输入尺寸：512×768；
- 最终配置：`configs/components_hku_adapted.yaml`；
- 最终权重：`runs/components_hku_adapted/best.pt`；
- BFDD RGB 采样权重：5.0；
- 微调：10 epochs，学习率 `1e-5`；
- 增强：50% 概率、±10° 小角度旋转，以模拟无人机姿态变化；不采用上下颠倒的垂直翻转。

既有纯 3DOM、clean sources、all sources 等实验及权重均被保留，没有覆盖。

### 3.2 IR 缺陷分割

- 模型：DeepLabV3+–ResNet50；
- 类别数：6（背景 + 五类缺陷）；
- 输入尺寸：512×640；
- 配置：`configs/ir_bfdd_weighted.yaml`；
- 权重：`runs/ir_bfdd_weighted/best.pt`；
- 训练：50 epochs，batch size 4，学习率 `1e-4`；
- 类别权重：`[1, 7, 8, 8, 7, 6]`。

IR 模型为独立训练任务，没有假定或复用 RGB 构件分割权重。ResNet50 的通用预训练初始化不等同于把已有 RGB 分割模型直接用于 IR。

### 3.3 空调二分类分割

- 模型：SegFormer-B0；
- 类别：background / air_conditioner；
- 配置：`configs/bfdd_ac_hku_adapted.yaml`；
- 权重：`runs/bfdd_ac_hku_adapted/best.pt`；
- 最终概率阈值：0.78。

后处理使用高置信核心与局部低阈值扩展，并只删除贴边的狭窄伪影。此前“删除所有贴左右边界连通域”的规则会误删贴边空调，现已替换为最大宽度 32 像素的窄侧边伪影过滤。

## 4. 测试结果

### 4.1 RGB 构件模型

最终 BFDD 适配模型在包含 BFDD RGB 的 203 张测试集上：

| 类别 | IoU | Dice |
|---|---:|---:|
| building_surface | 0.9004 | 0.9476 |
| opening | 0.6900 | 0.8166 |
| other | 0.8899 | 0.9417 |
| **Mean IoU** | **0.8268** | — |

在不含 BFDD 适配样本的 122 张 clean-source 测试集上，Mean IoU 为 0.7793。与 BFDD 适配前相比，目标域结果明显改善，但 `other` 类在 clean sources 上有所回落，因此保留此前模型作为对照基线。

加入 HKUST 小规模目标域适配后，8 张 HKUST validation 上的 Mean IoU 从 0.7431 提升到 0.8608，其中 `opening` IoU 从 0.5249 提升到 0.8337。新模型在原 203 张测试集上的 Mean IoU 为 0.8289，在 122 张 clean-source 测试集上为 0.7847，均未出现旧域退化。因此 HKUST 适配模型取代 BFDD 适配模型成为当前推荐权重，旧权重继续作为可复现基线保留。

### 4.2 IR 缺陷模型

152 张 BFDD 测试集结果：

| 类别 | IoU | Dice |
|---|---:|---:|
| Background | 0.9254 | 0.9613 |
| Cracks | 0.0384 | 0.0740 |
| Peeling | 0.0033 | 0.0066 |
| Hollow Areas | 0.2106 | 0.3480 |
| Stains | 0.2703 | 0.4256 |
| Erosion | 0.0816 | 0.1508 |
| **Mean IoU** | **0.2550** | — |

Cracks 和 Peeling 的结果很弱，这与热红外对细裂纹、表面颜色或纹理型剥落不敏感的成像机理一致。当前模型更适合捕捉具有明显热异常的 Hollow Areas、Stains 和部分 Erosion，不能把五类结果解释成同等可靠。

### 4.3 空调模型

旧模型固定阈值 0.63，在 4 张独立测试图上的像素指标为：

| 指标 | 数值 |
|---|---:|
| IoU | 0.7336 |
| Dice | 0.8463 |
| Precision | 0.8786 |
| Recall | 0.8163 |
| Background FPR | 0.0127 |

测试样本数量较少，因此该结果只证明当前标注域内可用，不能作为广泛泛化能力的充分证据。实际复核还发现水管、平台等可能被识别为设备；在当前用途下，这类结果作为“附属设备影响区”通常比误当墙面缺陷更安全。

加入 HKUST 正样本和窗框困难负样本后，混合 validation 选择阈值 0.78。新模型在原 4 张 BFDD test 上固定阈值 0.78 的结果为：IoU 0.7391、Dice 0.8500、Precision 0.9459、Recall 0.7717、Background FPR 0.0050。相较旧模型，精度和误报控制提高，召回率略有降低；这符合设备分支只提供保守参考提示的用途。HKUST validation 仅有 1 张空调正样本，因此跨域指标仍需谨慎解释。

## 5. RGB–IR 配准

使用 MFDN 的 `UST_Campus.pth` 提取 4096 个跨模态特征，并用 RANSAC 单应矩阵把 RGB 坐标映射到 IR 坐标。

全量结果：

- 输入 RGB–IR 对：838；
- 特征文件：838；
- 单应矩阵：838；
- 配准棋盘格：838；
- 人工总览复核：已完成，未发现需要整体排除的系统性失败。

单应矩阵只能描述近似平面变换。在窗框、突出立柱、空调和近景遮挡物存在显著深度差时，局部视差仍可能出现；因此后续三维投影需要多视图投票和边缘容差，不能把单张局部边界视为绝对精确。

## 6. 最终二维融合策略

最终固定参数：

| 参数 | 数值 |
|---|---:|
| AC threshold | 0.78 |
| AC min component area | 500 px |
| Narrow side-border width | 32 RGB px |
| AC local low threshold | 0.40 |
| AC local radius | 24 RGB px |
| AC influence radius | 24 IR px |
| Component edge uncertainty radius | 8 IR px |

缺陷状态定义：

| ID | 状态 | 可视化 | 含义 |
|---:|---|---|---|
| 0 | none | 无叠加 | 未预测为缺陷 |
| 1 | confirmed | 红色 | 位于墙面核心的缺陷预测 |
| 2 | uncertain | 橙色 | 位于构件边界或线性伪影复核区 |
| 3 | equipment_influenced | 紫色 | 缺陷预测与设备影响区重叠 |
| 4 | excluded_nonwall | 灰色 | 位于门窗、背景等明确非墙面区域 |

`defect_ids.png` 始终保留原始 IR 类别预测；过滤过程只生成状态层和派生结果，不覆盖原始预测。设备区域也不被直接抹除，而是单独标记，便于后续人工复核和三维展示。内部状态名 `excluded_nonwall` 仅表示“与参考非墙面区域重叠”，不授权删除数据；最终产品语义应展示为 `nonwall_overlap` 或等价的人工复核提示。

## 7. Dataset_1x 全量运行结果

838 张图像均成功生成：

- `summary.json`：838；
- `registration_checker.png`：838；
- `final_status_overlay.png`：838。

全量状态像素汇总：

| 状态 | 像素数 | 出现该状态的图像数 |
|---|---:|---:|
| confirmed | 23,001,083 | 781 |
| uncertain | 1,395,669 | 640 |
| equipment_influenced | 710,015 | 229 |
| excluded_nonwall | 4,591,142 | 488 |

其他统计：

- 原始缺陷预测共 29,697,909 像素，801 张图像存在非背景预测；
- 平均墙面覆盖率约 74.52%，中位数约 83.82%；
- 322 张图像存在非空设备 mask；平均设备覆盖率约 2.11%。

这些是模型输出规模统计，不是新的准确率指标；同一建筑被多帧拍摄，像素数也不能直接换算为真实缺陷面积。

## 8. 已知局限与使用边界

1. IR 五类缺陷性能不均衡，Cracks 和 Peeling 不应作为自动判定的主要依据；
2. 热成像伪彩红色不等于缺陷，必须区分底图颜色与状态叠加；
3. RGB 构件边界和单应配准存在局部误差，边缘状态需要保守处理；
4. 空调模型本质上是小样本设备分割器，可能把水管、平台或窗框局部识别为设备；
5. 当前 `confirmed` 表示通过二维空间规则的确认结果，不等于工程诊断意义上的最终缺陷结论；
6. 三维投影前必须保留来源图像、类别、状态和置信信息，不能只输出一个不可追溯的合并标签；
7. 多视图中的重复观测需要在三维空间聚合，避免把同一缺陷重复计数。

## 9. COLMAP 局部三维投影验证

由于 Dataset_1x 并非围绕同一栋建筑连续采集，完整图像无法形成单一、连续的 COLMAP 模型。本阶段不强行合并彼此独立的稀疏子模型，而是选择内部连通性较好的局部建筑片段，验证 RGB、IR、语义状态与三维点云之间的完整功能链。

### 9.1 局部重建

选用 2025-06-24 图像组的 COLMAP 子模型 8：

- 局部注册图像：27 张；
- 稀疏点：6,280；
- 平均重投影误差：约 0.611 px；
- 稠密融合点：48,143；
- 具有有效可见性记录的点：48,112。

该点云主要表现为建筑立面片段，接近平面是当前拍摄几何和局部建筑表面的自然结果，并不表示投影失败。

### 9.2 三维状态投票

对每个稠密点执行相机投影、深度一致性检查、RGB→IR 单应映射和二维状态采样。严格结果要求：

- 至少获得 4 个有效视角支持；
- 支持比例至少为 0.5；
- 深度容差为 0.02（COLMAP 模型坐标单位）。

严格点云结果为：

| 三维状态 | 点数 |
|---|---:|
| confirmed | 2,299 |
| uncertain | 3 |
| equipment_influenced | 0 |
| excluded_nonwall | 0 |

可确定类别的点包括 Cracks 113、Stains 770、Erosion 153；Peeling 和 Hollow Areas 在严格阈值下没有留下可靠三维点。状态点数和类别点数不能直接对应，因为状态投票与缺陷类别投票分别计算，部分类别未形成足够一致的多数结果。

### 9.3 3D→2D 回投影核查

将严格三维点重新投影到 8 个高覆盖视角，并与 RGB、IR 和原始二维融合状态并排核查。单视图一致率分别约为：99.72%、99.22%、98.74%、98.58%、92.95%、91.62%、83.16% 和 57.71%。其中较低结果表示多视图投票结论与该单帧预测不完全一致，不等同于几何配准失败。

人工侧视检查显示，着色缺陷点位于原始稠密点云的表面厚度范围内，没有形成脱离墙体的平行点层，也未发现系统性的背面投影。由于输出方式是在原始 `fused.ply` 点上写入状态和类别颜色，而不是另外生成带偏移的三维坐标，因此缺陷点与建筑点共享相同的几何位置。

本次验证证明以下局部链路可运行：

`RGB 构件分割 → IR 缺陷分割 → RGB–IR 配准 → 二维状态融合 → COLMAP 可见性与深度检查 → 多视图投票 → 三维缺陷着色`

该结论验证的是几何链路和数据接口；它不会消除 IR 模型本身的分类误差，也不能把模型预测直接解释为工程鉴定结论。

## 10. 后续工作

1. 在具备连续建筑航拍数据时，重建覆盖完整外立面的单一 COLMAP 模型；
2. 接入 FAST-LIVO2 位姿和点云，完成相机—LiDAR 外参及时间同步；
3. 将三维点的来源帧、支持视角数、状态比例和类别票数保存为可查询属性；
4. 增加跨视角空间聚类，形成缺陷区域而非仅展示着色点；
5. 使用现场复核数据评估三维定位误差及缺陷级 precision/recall；
6. 将 Cracks 和 Peeling 设为低可信提示，重点使用具有稳定热异常的类别。

点云离群点清理和展示美化不属于当前功能验证的必要条件，本阶段按人工确认跳过。

## 11. 当前可复现入口

- 构件配置：`configs/components_hku_adapted.yaml`
- 构件权重：`runs/components_hku_adapted/best.pt`
- IR 配置：`configs/ir_bfdd_weighted.yaml`
- IR 权重：`runs/ir_bfdd_weighted/best.pt`
- 空调配置：`configs/bfdd_ac_hku_adapted.yaml`
- 空调权重：`runs/bfdd_ac_hku_adapted/best.pt`
- 空调参考阈值：`0.78`
- HKUST 适配数据：`data/hku_adaptation/prepared/`
- MFDN 矩阵：`MFDN/results/BFDD_All/homo_ir2vis/reproj/4096/`
- 全量融合结果：`runs/fusion_status_dataset1x/`
- COLMAP 局部投影程序：`seg_tasks/project_colmap_fragment.py`
- 严格三维投影结果：`runs/colmap_fragment_8_strict/`
- 3D→2D 核查程序：`seg_tasks/audit_colmap_projection.py`
- 回投影核查总览：`runs/colmap_fragment_8_projection_audit/contact_sheet.jpg`
- 融合实现：`seg_tasks/inspect_pair.py`

root dir在projects/InfraDetect/IR.
