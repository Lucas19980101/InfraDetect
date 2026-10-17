# Thermal-only 独立热图对照分支

- **文件**：`checkpoint.pt`；来自原实验
  `IR/runs/thermal_tbbr_segformer_b2_native_patches_ade_init/best.pt`。
  验证集 best 为第 40 epoch，原训练计划共 50 epoch。
- **功能**：只根据热图输出两类概率，可读作
  `Non-thermal-anomaly / Thermal anomaly`（未检出热异常 / 热异常候选）。
  配置和权重中的原始类名仍是 `Background / Thermal Anomaly`；
  `Background` 不等于人工确认的正常墙面。用于与
  RGB+Thermal 主模型做模型差异和相对独立的模态证据核对；不是第二种缺陷真值。
- **网络**：SegFormer-B2，ADE20K 权重初始化，再用 TBBR 热图训练；**无 TASeg
  初始化，也不读取 RGB**。这与主分支的初始化和输入不同。

## 输入和输出

标准输入是 `H×W×4` `[B,G,R,Thermal]` 场景栈，但仅读取第 4 通道；
旧五通道栈也兼容，附加通道不参与推理。
按 `(value-125)/34` 归一化后重复成模型需要的 3 通道。输出是原图 `H×W`
热异常概率 `thermal_probability.npy`，以及供模型差异查看的二值候选。
RGB/height 不参与模型判断。

## 数据与训练方法

与主分支共用 TBBRv2 航线级划分：664 train、59 Flug1_103 val、203
Flug1_105 test。训练为原始像素 `1024×1280` 随机裁块，单图一块，正样本
裁块概率 0.75；batch 1、梯度累积 4、50 epoch、类别权重 `[1,8]`，
以 val 异常 IoU 选择 best。配置见
同目录的 `config.yaml`。
best 验证异常 IoU 约 **0.3926**。

## 已有表现与边界

固定 Flug1_105 test（203 张，原生分辨率滑窗）：异常 IoU **0.3537**、
Dice **0.5226**、Precision **0.5568**、Recall **0.4924**；含背景 mIoU
**0.6735**。来源：原实验 `test_best/metrics.json`。与主分支相比，这一测试
划分上异常 IoU/Recall 较高、Precision 较低；不代表目标立面必然更好。
该数据集的标注是热桥，不能验证空鼓或污渍分类。

## 推理方法

`infradetect.evidence.export_thermal` 使用本权重和 YAML，按原始像素
`1024×1280`、25% 重叠滑窗推理，logits 均值融合后 softmax，还原完整
`H×W` 概率。综合入口把它放在 `thermal_diagnostic/`，并生成与主分支的差异图。
