# RGB + Thermal 热异常主分支

- **文件**：`checkpoint.pt`；来自原实验
  `IR/runs/thermal_tbbr_segformer_b2_rgb_thermal_fusion_fullfinetune/best.pt`。
  验证集选出的 best 为第 40 epoch；原训练计划共 50 epoch。
- **功能**：输出逐像素“热异常候选”概率，供二维融合和人工复核。TBBR 真值语义是
  *thermal bridge*，并非空鼓、污渍或已经确认的建筑病害。
- **网络**：双分支 SegFormer-B2。RGB 分支以 ADE20K 预训练参数初始化；热分支从
  TASeg 预训练后再在 TBBR 训练的模型初始化。RGB 分支也参与微调，学习率为主分支
  的 0.1 倍。**仅这个热异常分割头**输出两类，可读作
  `Non-thermal-anomaly / Thermal anomaly`（未检出热异常 / 热异常候选）。
  权重和配置中的原始类名仍是 `Background / Thermal Anomaly`；其中
  `Background` 只表示模型的非异常类，**不等于**人工确认的正常墙面。

## 输入和输出

输入为已配准的 `H×W×4` NumPy 数组，通道 `[B,G,R,Thermal]`；旧五通道
场景也可读，但 Height 不进入本模型。本模型
实际使用 RGB 与第 4 个 thermal 通道，不使用 height。RGB 按 ImageNet 参数归一化；
thermal 使用 `(value-125)/34`，重复为 3 通道后与 RGB 合成 6 通道网络输入。
输出为原图网格 `H×W` 的热异常概率 `thermal_probability.npy`，并在融合阶段按
清单中的阈值转成候选区域。RGB 与 thermal 必须经过可靠配准。

## 为什么系统里还会看到“正常”或“干扰”

热异常主模型**并没有** `Normal Wall`、`Air-conditioner Interference`、
`Hollow` 等输出类。它只回答“这里像不像训练标签中的热异常”。
综合系统把其他证据放在这张概率图旁边，才产生更细的复核提示：

- **非热异常类（原名 Background）**：热模型未报异常的像素；若 ROI 外、无效或未配准，
  只能记为“未评估/无法对应”，不能叫已确认正常。
- **热干扰线索**：独立的 RGB 空调模型和构件/墙面模型提示设备、
  非墙面或边界；它们不把热概率改成“正常”，也不保证找出日照、
  反射等所有干扰。
- **人工复核状态**：融合后的热候选可能被标为墙面支持、设备影响、
  配准/边界不确定等，用于解释和排序；这些是规则与证据状态，
  **不是**热模型新增的训练类别。最终是否为真实缺陷仍需人工核对。

独立 IR-only 热模型也只有相同的两类输出。RGB 可见缺陷模型则另外输出
六个可重叠类别；不能把这六类、热模型两类和融合复核状态混成
一个“多分类模型”。

## 数据与训练方法

- TBBRv2 航线级划分：664 train（Flug1_100/101/102/104）、59 val
  （Flug1_103）、203 固定 test（Flug1_105）。test 不用于选 epoch 或阈值。
- 训练使用原始像素的 `1024×1280` 随机裁块，单图一块，正样本裁块概率 0.75；
  **不是**每轮对整张图做滑窗训练。batch 1、梯度累积 4、50 epoch，类别权重
  `[1,8]`，优化器/学习率等见同目录的 `config.yaml`。
- best 按 Flug1_103 验证集的异常 IoU 选择；保存的 best 指标约 **0.4705**。

## 已有表现与边界

对固定 Flug1_105 的 203 张图，以原生分辨率滑窗评估：异常 IoU **0.3365**、
Dice **0.5036**、Precision **0.6733**、Recall **0.4022**；含背景的 mIoU
**0.6653**。来源：原实验 `test_best/metrics.json`。此模型精度并未全面超过
Thermal-only 对照：它在该测试集的异常精确率较高，但召回较低。

这些数值只反映 TBBR 屋顶/航拍热桥标签，不代表立面热异常泛化能力，更不是
“空鼓检测准确率”。主分支已经读取 RGB，不能将其与另一个 RGB 缺陷模型的重叠
称为独立模态验证。

## 推理方法

`infradetect.evidence.export_thermal` 加载本权重及上述 YAML，调用
`infradetect.inference.thermal.predict_full` 在原始图上以 `1024×1280`、25% 重叠滑窗推理，
重叠处先平均 logits，再做 softmax；不缩小整张图。完整入口由
`python -m infradetect_prototype run --manifest ... --output ...` 调用。
简单说，滑窗只保证**送进模型前不把整张大图压小**：原图上的小区域会以
原来的像素大小进入模型。但 SegFormer 在计算过程中仍会把图像变成较小的
内部特征图，再恢复为原图尺寸的输出。因此，输出虽然是完整的 `H×W`
概率图，也不代表每一处细小纹理都被完整保留；很细或很弱的异常仍可能漏检。
