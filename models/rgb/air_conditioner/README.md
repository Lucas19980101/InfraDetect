# RGB 空调设备 / 热干扰提示模型

- **文件**：`checkpoint.pt`；来自原实验 `IR/runs/bfdd_ac_hku_adapted/best.pt`，
  验证集选出的 best 为第 2 epoch（配置计划 8 epoch）。
- **功能**：从 RGB 图定位可能的空调设备，给热异常区域添加“设备影响”提示。
  **没有检测到空调不等于排除了日照、反射、材质差异或其他热干扰。**
- **网络**：SegFormer-B0，`background / air_conditioner` 二类，
  以 `nvidia/mit-b0` 初始化后适配 BFDD 与 HKUST 场景。

## 输入和输出

输入为 RGB 三通道图像（四通道起的场景栈中取 BGR 第 1–3 通道，转 RGB 后 ImageNet
归一化）。输出为原图 `H×W` 的空调类别概率，综合入口保存
`rgb/interference_probability_rgb.npy`，再按已核验 RGB→IR 变换投到热图网格。
其概率只针对“像空调设备”这一类，不是完整热源/干扰概率。

## 数据与训练方法

合并 BFDD 原图、空调正/负裁块和 HKUST 空调标注；manifest 为 92 train、
24 val、4 test。模型训练尺寸 `512×768`、batch 2、计划 8 epoch、学习率
`5e-6`、weight decay 0.01、±10° 旋转；类别权重 `[0.5,1.0]`，HKUST 样本
加权采样。配置：同目录的 `config.yaml`。

## 已有表现与边界

- BFDD 原始来源 test **仅 4 张**：空调 IoU **0.6957**、Dice **0.8205**，
  含背景 mIoU **0.8282**。
- HKUST 适配 val **8 张**：空调 IoU **0.0244**、Dice **0.0476**，
  含背景 mIoU **0.4897**。

来源：原实验 `test_original_bfdd/metrics.json`、`hku_val_after/metrics.json`。
跨域表现明显不稳定，不能将此分支当作可靠的干扰过滤器，尤其不能因为它没响应
就提升病害置信度。上述数值也来自固定尺寸评估；交付版原像素裁块的同口径
系统指标尚未建立。

## 推理方法

`infradetect.evidence.export_rgb` 在原图使用 `512×768` 重叠裁块，对 softmax
概率做均值融合，输出完整 H×W 空调概率图，不缩小整张图。融合规则只添加
“可能受设备影响”的上下文状态，不应把空调区域等同于所有热异常的原因。
