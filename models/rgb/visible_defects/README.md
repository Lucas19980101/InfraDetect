# RGB 六类可见缺陷模型

## 交付的是哪份权重

`checkpoint.pt` 是 **Experiment B / `segformer_b0_native_classaware_v1`** 的推理版，
来自该实验第 36 epoch 的 `best.pt`。网络是 SegFormer-B0，输出六个可重叠的
sigmoid 通道：`crack`、`spalling`、`exposed_rebar`、`rust`、
`efflorescence`、`wetspot`。它找的是 RGB 可见证据，不是热异常或最终病害确诊。

为便于交付，推理版只保留 `model/config/epoch/thresholds` 等必要信息；
208 个模型权重张量与原 `best.pt` 逐一相同。原完整 checkpoint 的 SHA256 是
`f880667fe3d1d87993c02e64358b7dbaf8a8958d66dd14c4baf3970ac8d9c4fd`；
本目录推理版的 SHA256 见 [`../../SHA256SUMS`](../../SHA256SUMS)。推理版没有
optimizer、scheduler、scaler，不能原样续训。原完整 checkpoint 不随包交付。

## 输入、处理和输出

输入是 RGB 三通道原图（程序从 BGR 转换，再用 ImageNet 均值/标准差归一化）。
程序在**原像素尺度**上以 512×512 patch、384 像素 stride 滑窗，不先把整图
缩到 512×512。重叠位置的 sigmoid 概率取平均，得到原图大小的六通道概率图
`rgb/rgb_semantic_probabilities.npz`、阈值位掩码、`coverage` 和每块的原图坐标。
可选墙面 ROI 只控制参与计算的 patch；未覆盖区域是“未评估”，不是阴性。

随权重保存的六类阈值按上述顺序为 **0.705、0.858、0.621、0.498、0.753、
0.859**。阈值用于生成候选标记；概率图仍保留，人工复核时不应只看二值结果。

## 训练设置与思路

同目录的 [`training_config.yaml`](training_config.yaml) 是当前交付的 B 权重所用
配置快照，逐项注明了原始参数与数据路径的含义。它是**训练记录**，不是推理
入口读取的配置；交付包不含训练代码、数据或初始化用的 A 权重。

这轮实验固定模型和 DACL10K 划分，主要比较采样协议：

1. **A（起点）**：`segformer_b0_nativecrop_v2`，从旧六类模型初始化；
   在原图随机裁 512×512，小图 pad 而不缩放。验证用原图滑窗与概率平均。
2. **B（当前交付）**：从 A 的最佳权重初始化，仍按原尺度裁片，但按类别
   有倾向地采正例，同时保留 10% 背景裁片。六类采样权重依次是
   22%、10%、18%、8%、12%、20%；正例裁片至少有 16 个有效正像素。
   这样做是为了增加稀有缺陷被看到的机会，不改变六类输出结构。
3. **C70（对照，未交付）**：从 B 初始化，训练裁片来源为 DACL 70%、
   DamCrack train 15%、CUBIT crack train 7.5%、CUBIT spalling train 7.5%。
   外部数据只监督自身确有标签的类别；其他类别在 BCE **和 Dice** 中均屏蔽，
   不把“没标注”当成“没有缺陷”。C70 没有取得更高的总体 DACL 验证分数，
   所以没有替换 B。

B 的记录设置：seed 42；最多 50 epoch，最佳为 epoch 36；AdamW、
CosineAnnealingLR、学习率 `1e-5`、weight decay `0.01`、batch size 4、
梯度累积 2、AMP；有效像素上的 `0.5 × masked BCE + 0.5 × masked Dice`。
每 2 epoch 验证，连续 10 次验证无提升则提前停止。验证 patch 512、stride 384，
重叠概率平均；没有开启 wall ROI。保存验证集六类平均 IoU 最优的 checkpoint，
六类阈值在同一验证集校准。

数据划分清单记录：DACL train 6241、val 694、test 975、unlabeled 1012；
实际可评估 val 为 **693 张**，另 1 张标签全无效而排除。B 的训练和选模
只使用 DACL train/val，**未使用 DACL test**。C70 的 DamCrack 和 CUBIT
只取 train，外部 val/test 未参与训练或选模；这些是实验记录，不代表已完成
跨数据集去重证明。训练数据、代码及完整日志不包含在本推理交付包内。

## 已验证表现与边界

同一 DACL 原图滑窗验证协议下，A/B/C70 六类平均 IoU 分别为 **0.39818 /
0.40621 / 0.40392**。B 的单类 IoU 按上述顺序为 **0.42549、0.40691、
0.38441、0.55951、0.46496、0.19595**。B 相对 A 主要提升 exposed_rebar，
但 wetspot 下降；C70 的 crack 也没有继续提升。上述数字均来自同一 DACL
val，且阈值在 val 上校准，**不是独立 test、目标建筑或综合系统准确率**。
即使候选位置有用，像素分割也可能不精确，仍需在 RGB/IR 与三维界面复核。

## 推理调用

`infradetect/evidence/export_rgb.py` 调用
`infradetect.inference.rgb_predict.predict_rgb_defects`。
场景清单中权重路径仍是 `models/rgb/visible_defects/checkpoint.pt`，无需修改入口。
