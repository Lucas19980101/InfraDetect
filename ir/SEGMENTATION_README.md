# InfraDetect 的两项独立分割训练

这里的代码在 `IR/seg_tasks` 中。`DeepLabs+` 的 RGB 实验与 checkpoint 不参与 IR 训练；IR 模型仅采用同类 DeepLabV3+–ResNet50 架构和 CE+Dice 训练方式。构件模型参考 Hugging Face SegFormer-B0 的语义分割接口。

## 已核对的数据

- `collected_datasets/Dataset_1x`：838 组同名 RGB JPG、伪彩 IR PNG、单通道 `Label` PNG。当前 `Label` 只含 `0–5`；`train.txt` 有 686 张、`test.txt` 有 152 张。数据提供者确认的映射是 `0=Background, 1=Cracks, 2=Peeling, 3=Hollow Areas, 4=Stains, 5=Erosion`；RGB palette 分别为 `(0,0,0)`, `(61,61,245)`, `(169,36,191)`, `(174,79,13)`, `(36,179,83)`, `(203,253,0)`。`Label_backup_7classes_20260125` 含 `0–6`，其中旧值 `3` 的 18,652 个像素并入当前值 `4`；原 `Label_color` 在这些像素上保留旧颜色，训练以单通道 `Label` 为准。
- `collected_datasets/3DOM`：227 张 RGB 图像及标签。`labels_gray` 实际是三个通道相同的 PNG，值 `1–5` 对应 facade、window、obstacle、sky、door；`0` 是混合边界，转换为 `255=ignore`。七个不规则文件名已显式配对。另有 51 对图像与标签宽高对调；逐对边界核验均支持把 mask 逆时针旋转 90°，读取器会自动处理，原始文件不变。
- 数据审查图：`audit_outputs/bfdd_ir_labels_contact_sheet.jpg`。它展示五种当前 BFDD 标签的代表性 RGB、IR 和彩色标注。

## 划分

生成并检查清单：

```bash
python -m seg_tasks.splits --task ir --root ../collected_datasets/Dataset_1x --output data/bfdd_split.csv
python -m seg_tasks.splits --task components --root ../collected_datasets/3DOM --output data/3dom_split.csv
```

BFDD 保留原 `test.txt`，从 train 中选取完整的 30 秒采集段作 val，得到 581/105/152。原 train/test 存在相邻时间帧（15 张 test 图与 train 图相隔不超过 30 秒），因此这个 test 仍可能高估跨场景泛化能力。3DOM 按保守的文件名家族分组，得到 183/22/22；文件名不能完全证明建筑身份，正式跨建筑评估前应人工核查清单中的 `group`。

## 训练

从本目录 `IR/` 运行，使用已经装有 PyTorch、Transformers、SMP、OpenCV 和 PyYAML 的环境：

```bash
python -m seg_tasks.train --config configs/components_3dom.yaml
python -m seg_tasks.train --config configs/ir_bfdd.yaml
python -m seg_tasks.train --config configs/ir_bfdd_weighted.yaml
```

三分类构件模型采用分阶段训练，并保留原五分类实验。先生成固定的分组划分：

```bash
python -m seg_tasks.split_components_3class
```

阶段二使用 3DOM、CMP base/extended 和 eTRIMS-8，共 649 train / 122 val / 122 test。以旧 3DOM 五分类最佳模型初始化形状兼容的 206 个张量，只重新初始化三分类输出层：

```bash
python -m seg_tasks.train \
  --config configs/components_clean_3class.yaml \
  --init-checkpoint runs/components_3dom/best.pt
```

训练结果写入 `runs/components_clean_3class/`，不会覆盖 `runs/components_3dom/`。阶段三需等 Hznu / perspective correction 的最终纳入清单确定后再运行；当前 `components_all_3class.yaml` 只包含清洗后的 Hznu original，不包含 perspective correction。

BFDD RGB 人工修正草稿确认后，第四阶段域适配先构建组合 manifest：

```bash
python -m seg_tasks.prepare_bfdd_component_adaptation
```

66 张按原拍摄组整体划分为 48 train / 10 val / 8 test，同组不跨 split。训练仍保留 3DOM、CMP、eTRIMS 和 Hznu；BFDD 训练样本使用 5 倍采样权重（约占每轮 20%），配置为 `components_bfdd_adapted.yaml`，从 `components_all_3class/best.pt` 以 `1e-5` 学习率微调 10 轮。草稿标签是粗粒度域适配标签，不应被描述为像素级精确真值。

预训练来源分别是 `nvidia/mit-b0` 的 MiT-B0 编码器和 ImageNet ResNet50 编码器；两项任务的分割头从头训练。第一次执行可能需要下载通用编码器权重。训练结果分别写入 `runs/components_3dom/`、`runs/ir_bfdd/` 和 `runs/ir_bfdd_weighted/`，互不覆盖。训练会打乱样本并丢弃不足一个完整 batch 的尾批，以避免 DeepLab 的 BatchNorm 收到单张图像。CPU 可用于数据和小规模功能检查；正式训练建议有可用 CUDA。

BFDD 配置已使用数据提供者确认的 ID 顺序。公开 BFDD 页面描述的是 788 对图像，本地目录有 838 对；本工程以本地 `Label`、所给映射和本地划分为准。模型的六个输出通道与上述 ID 一一对应。评估同时保存单通道类别 PNG 和 `color/` 下按 palette 着色的预览 PNG；后续配准、过滤和三维投影只使用单通道类别图。

## 评估

```bash
python -m seg_tasks.evaluate --config configs/components_3dom.yaml --checkpoint runs/components_3dom/best.pt --split test --output runs/components_3dom/test
python -m seg_tasks.evaluate --config configs/ir_bfdd.yaml --checkpoint runs/ir_bfdd/best.pt --split test --output runs/ir_bfdd/test
```

输出包括逐类 IoU、Dice、mIoU、混淆矩阵和单通道类别 PNG。PNG 恢复到输入图像的原始分辨率；配准与三维投影应使用这个类别图，不能使用彩色预览图。

## 配准后过滤

`seg_tasks.filter.filter_defects` 接收 IR 缺陷类别图、RGB 构件类别图，以及一个 **RGB 像素到 IR 像素** 的 3×3 单应矩阵。它用最近邻插值把构件 0 类 facade 转到 IR 平面，返回过滤后的缺陷图和墙面允许区。MFDN 的 `reproj_only.py` 可由匹配点估计矩阵；其 `homo_ir2vis` 目录名沿用旧名，但实际存储的 `H` 是 RGB 到 IR 方向。应用前仍需检查棋盘叠加图和配准误差。

```bash
python -m seg_tasks.filter --defect-mask ir_prediction.png --component-mask rgb_prediction.png \
  --rgb-to-ir H.mat --output filtered.png --wall-output wall_in_ir.png
```

配准不可靠时，函数调用可传入 `valid_registration=False`，保留原始缺陷图供人工检查。正式评估应同时报告过滤前后指标以及被过滤的真缺陷数量。

## 当前训练状态

两项模型均完成 50 轮训练，并已各评估一次 test。下列数字是已有运行结果，不代表跨建筑部署性能：

| 实验 | 已运行轮数 | 验证集最佳 mIoU | test mIoU | 备注 |
|---|---:|---:|---:|---|
| `components_3dom` | 50 / 50 | 0.7028（第 34 轮） | 0.5059 | 墙面 IoU 0.8946；门 IoU 0.0444 |
| `ir_bfdd_weighted` | 50 / 50 | 0.2852（第 28 轮） | 0.2550 | 缺陷五类平均 IoU 0.1209；Cracks 0.0384，Peeling 0.0033 |

`audit_outputs/ir_weighted_epoch4_samples.jpg` 是早期验证样本，最终结果以 `runs/*/test/metrics.json` 为准。当前 IR 模型不能作为可靠的五类缺陷自动标注器，尤其是 Cracks 与 Peeling。

三维投影仍需经过验证的逐图 RGB→IR 配准矩阵和相机位姿；目前尚未执行。

## 成对图片推理与过滤检查

先在同一场景的 RGB、IR 图像上分别用当前 `best.pt` 推理。BFDD 的 `RGB/` 与 `IR/` 同名图片可用于模型输出检查；**它们是否已逐像素对齐不能仅凭文件名认定**。不提供矩阵时，脚本只保存 `component_ids.png`、`defect_ids.png` 及原图叠图，不执行墙面过滤。

```bash
python -m seg_tasks.inspect_pair \
  --rgb ../collected_datasets/Dataset_1x/RGB/DJI_20250624181809_0003.JPG \
  --ir ../collected_datasets/Dataset_1x/IR/DJI_20250624181809_0003.png \
  --ir-label ../collected_datasets/Dataset_1x/Label/DJI_20250624181809_0003.png \
  --output runs/pair_inspection/DJI_20250624181809_0003
```

拿到并**人工核验**该图片的 RGB 像素到 IR 像素 3×3 单应矩阵后，加入 `--rgb-to-ir 路径/H.mat` 或 `.npy` 重新运行。此时额外保存 `registration_checker.png`、`wall_in_ir.png`、`filtered_defect_ids.png`、`filtered_overlay.png` 和 `summary.json`。先打开棋盘叠图核验边缘；若配准不可靠，忽略过滤图。`--ir-label` 可选，提供时 `summary.json` 会记录过滤前后缺陷二值真阳性／误检像素，以及被删的正确类别像素数。没有标签时仅可视化，不能声称过滤提升了精度。

正式三分类融合使用 `components_bfdd_adapted/best.pt`。有经人工核验的 RGB→IR 矩阵后，`inspect_pair` 会输出墙面核心、边缘不确定带和明确排除区；默认边缘半径为 8 个 IR 像素。红色为确认缺陷、橙色为边缘候选、灰色为排除响应。边缘候选不会进入“确认缺陷”，但保留供复核。

所有 `*_ids.png` 均为原图尺寸的单通道类别 ID，供后续融合／投影使用；`*_overlay.png` 仅供目视检查。脚本不会写回原图、训练权重或既有测试输出。调过滤规则应使用验证集，测试集只用于固定方案的最终报告。

### BFDD 验证集上的构件域迁移检查

下面的命令按固定规则从不同采集组选择 10 张 BFDD `val` RGB 照片，只运行 3DOM 构件模型，不使用 BFDD `test`，也不假定 BFDD 有构件真值：

```bash
python -m seg_tasks.audit_components_bfdd --count 10 --output runs/component_bfdd_audit
```

打开 `contact_sheet.jpg` 和每张 `*_overlay.png`，重点人工检查真实外墙被预测为黄色 sky、红色 obstacle 等非 facade 的区域。`summary.json` 的面积比例只是模型预测统计，不能当成准确率。若墙面误删常见，不应直接启用硬墙面门控；先记录误分模式，再决定是否需要构件域适应或软过滤。
