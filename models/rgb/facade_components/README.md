# RGB 立面构件 / 墙面区域模型

- **文件**：`checkpoint.pt`；来自原实验 `IR/runs/components_hku_adapted/best.pt`，
  验证集选出的 best 为第 1 epoch（配置计划 8 epoch）。
- **功能**：给出 `building_surface / opening / other` 三类上下文，辅助判断热异常
  是否落在墙面核心、开口或其他物体上；它不是裂缝/空鼓等病害分类器。
- **网络**：SegFormer-B0，使用 `nvidia/mit-b0` 初始化并在多来源构件标签上适配。

## 输入和输出

输入是原图 RGB（若来自四通道起的场景栈，取第 1–3 通道 BGR，再转 RGB 并按 ImageNet
参数归一化）。输出为原图 `H×W` 的三类概率与类别 ID；综合入口保存
`rgb/component_ids_rgb.png`，后续通过已核验的 RGB→IR 变换投影为墙面/开口上下文。

## 数据与训练方法

训练集由 3DOM、CMP（base/extended）、eTRIMS8、HZNU、BFDD RGB 和 HKUST
适配样本组成；当前合并 manifest 有 1067 train、213 val、203 test，另 140
条 exclude。输入训练尺寸 `512×768`，batch 2、计划 8 epoch、学习率 `5e-6`、
weight decay 0.01，±10° 旋转增强；BFDD RGB/HKUST 训练样本加权采样。
配置：同目录的 `config.yaml`。

## 已有表现与边界

原始来源混合测试 203 张：mIoU **0.8289**；building_surface **0.9006**、
opening **0.6896**、other **0.8966**。另 HKUST 适配验证 8 张：mIoU
**0.8608**。来源：原实验 `test_original_sources/metrics.json`、
`hku_val_after/metrics.json`。

这些数值使用当时的固定尺寸评估流程；交付系统改用原始像素重叠裁块，**尚没有
同口径的完整重测指标**。多来源构件标签也不是建筑病害真值，墙面类别预测错误会
影响后续候选优先级；不应把低墙面概率直接当作“无缺陷”。

## 推理方法

`infradetect.evidence.export_rgb` 用 `infradetect.inference.rgb_predict.predict_native_class_probabilities`
以 `512×768` 原像素裁块、约 75% patch stride 推理，重叠区平均 softmax 概率，
再 argmax 为类别 ID。整个 H×W 原图不做全局缩小。综合入口会把它作为
**上下文证据**，并保留配准有效区域。
