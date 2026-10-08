# InfraDetect RGB 六通道多标签分割：项目交接报告

> 用途：将本文件上传到 ChatGPT，或在 ChatGPT 桌面客户端关联本地目录
> `/home/lucas19980101/projects/InfraDetect/RGB/6ML`。阅读本文件后再讨论项目，
> 不要假设能够访问此前 Codex 对话。

## 1. 项目目标与当前状态

本项目对 DeepLabV3+ 与 SegFormer-B0 进行公平的 RGB 多标签缺陷分割比较。
两个模型共用同一数据划分、输入尺寸、增强、损失、阈值搜索和评估器。

六个独立输出通道为：

1. `crack`
2. `spalling`
3. `exposed_rebar`
4. `rust`
5. `efflorescence`
6. `wetspot`

模型输出为 `[B, 6, H, W]` logits，每通道独立使用 sigmoid。没有 background
输出通道；背景由六通道全阴性推导。禁止使用七分类 softmax，因为缺陷可以重叠。

目前已经完成：

- DACL10K 下载、完整性检查和标签审计；
- 六通道标签映射与其他类别 ignore mask；
- 固定 train/val/test 协议；
- DeepLabV3+-6ML 和 SegFormer-B0-6ML 训练；
- DACL10K 公平测试；
- 当前建筑 crack/spalling partial-label 微调；
- 当前建筑测试和 DACL10K 遗忘检查；
- 14 项自动测试全部通过。

## 2. 重要目录

统一六通道工程：

```text
/home/lucas19980101/projects/InfraDetect/RGB/6ML
```

DACL10K：

```text
/home/lucas19980101/projects/InfraDetect/RGB/datasets/dacl10k
```

当前建筑 crack/spalling 数据及原三分类工程：

```text
/home/lucas19980101/projects/InfraDetect/DeepLabs+
```

旧官方 SegFormer/MMCV 仓库（仅参考，不用于当前训练）：

```text
/home/lucas19980101/projects/InfraDetect/RGB/SegFormer
```

已验收的三分类 DeepLabV3+ 位于 `../DeepLabs+`，输出为
background/crack/spalling softmax。该工程及 checkpoint 保持冻结，六通道开发没有覆盖它。

## 3. DACL10K 标签协议

映射关系：

```text
crack          = Crack union ACrack
spalling       = Spalling
exposed_rebar  = ExposedRebars
rust           = Rust
efflorescence  = Efflorescence
wetspot        = Wetspot
```

其余 DACL10K 类别不能当作背景。像素监督优先级：

```text
六类目标阳性 > 其他 DACL10K 类别 ignore > 未标注负背景
```

如果目标类与其他类重叠，目标阳性仍参与监督。仅属于其他类的像素不参与损失或指标。

标签缓存位于：

```text
RGB/datasets/dacl10k/cache/labels_6ml
```

缓存为单通道无损 PNG：bit 0–5 保存六类，bit 6 保存 valid mask。缓存共 7910
张，约 94 MB。原始 JSON 和图片没有修改。

## 4. 数据划分

DACL10K 公开的 `testdev` 没有标签，因此采用：

```text
官方 train 6935：6241 train + 694 experiment-val
官方 validation 975：frozen local test
官方 testdev 1012：仅无标签推理
```

划分对六类、21 个两两共现特征及无六类样本做确定性分层，随机种子 42。

manifest：

```text
6ML/data/split_manifest.csv
SHA-256: 9c868eeb8564914a4613482663598fd2bf0a98a0b8fa23ac97e24446c1ff94df
```

严禁重新随机划分后与现有结果直接比较。

## 5. 训练协议

两种模型共同使用：

```text
输入尺寸：整图 resize 到 512x512
batch size：4
梯度累积：2
有效 batch：8
epoch：50
优化器：AdamW
学习率：6e-5
weight decay：0.01
调度器：CosineAnnealingLR
混合精度：启用
损失：0.5 masked BCEWithLogits + 0.5 masked soft Dice
```

训练增强包括水平/垂直翻转、90 度旋转、轻微亮度对比度变化、轻微模糊或噪声、
小幅仿射。几何增强同步作用于图像、六类 mask 和 valid mask；颜色增强只作用于图像。
val/test 没有随机增强。

DeepLabV3+ 使用 ImageNet ResNet50 编码器；SegFormer 使用 ImageNet MiT-B0 编码器。
两者解码器和六通道头均针对任务训练。模型内部不调用 sigmoid。

## 6. 原始 DACL10K 训练结果

checkpoint：

```text
runs/deeplabv3plus_6ml/best.pt       # best epoch 47
runs/segformer_b0_6ml/best.pt        # best epoch 34
```

冻结 test 结果：

| 指标 | DeepLabV3+ | SegFormer-B0 |
|---|---:|---:|
| 六类 mIoU | 0.3994 | **0.4135** |
| crack IoU | 0.4789 | **0.4814** |
| spalling IoU | **0.4736** | 0.4485 |
| exposed rebar IoU | 0.3422 | **0.3893** |
| rust IoU | 0.4554 | **0.5049** |
| efflorescence IoU | **0.3761** | 0.3523 |
| wetspot IoU | 0.2704 | **0.3046** |
| 推理速度（batch 4） | 146.5 img/s | **276.9 img/s** |
| 峰值推理显存 | 955.7 MiB | **719.4 MiB** |
| 模型状态大小 | 102.0 MiB | **14.2 MiB** |

修正后的 relaxed crack F1 必须只在相同的 286 张含 crack 真值图片上平均：

```text
DeepLabV3+：0.4941
SegFormer-B0：0.4376
```

无 crack 真值图片的假阳性图像率：

```text
DeepLabV3+：18.58%
SegFormer-B0：22.21%
```

注意：早期 metrics JSON 中曾出现 relaxed F1 为 0.3414/0.2851，原因是分母随模型
预测变化，不能使用。评估器已修正。普通 IoU/Dice/precision/recall 不受影响。

通用六类主模型建议使用未微调 SegFormer：

```text
runs/segformer_b0_6ml/best.pt
```

## 7. 当前建筑 partial-label 微调

建筑数据：

```text
train: 4286
val:    840
test:   840
```

数据实际来自两个独立标注源，而不是每张图都完整标注两类：

```text
crack source：只监督 crack
spalling source：只监督 spalling
其余通道：全部屏蔽损失
```

微调配置刻意保守：冻结编码器、学习率 `1e-5`、5 epoch。未标注通道梯度测试为零。
原 DACL checkpoint 未被覆盖。

微调 checkpoint：

```text
runs/deeplabv3plus_6ml_building/best.pt
runs/segformer_b0_6ml_building/best.pt
```

建筑 test 结果（每个 checkpoint 均先在同一建筑 val 校准 crack/spalling 阈值）：

| 指标 | DeepLab 微调前 | DeepLab 微调后 | SegFormer 微调前 | SegFormer 微调后 |
|---|---:|---:|---:|---:|
| crack IoU | 0.3847 | **0.5077** | 0.2879 | 0.4850 |
| spalling IoU | 0.5211 | 0.5982 | 0.5680 | **0.6216** |
| 两类 mIoU | 0.4529 | 0.5530 | 0.4280 | **0.5533** |
| relaxed crack F1 | 0.6761 | **0.8255** | 0.5647 | 0.8217 |
| 无 crack 图假阳性率 | 5% | **15%** | 14.5% | 34% |

两个微调模型总体 mIoU 基本相同。当前建筑更推荐微调 DeepLab，因为 crack IoU 更高且
无 crack 图片误报显著更低；SegFormer 的 spalling 更好。

当前建筑推荐 checkpoint：

```text
runs/deeplabv3plus_6ml_building/best.pt
```

建筑阈值：

```text
crack: 0.413
spalling: 0.496
```

## 8. 微调后的 DACL 遗忘

先在 DACL experiment-val 重新校准阈值，再评估 frozen test：

| 模型 | 微调前 DACL mIoU | 微调后重新校准 | 下降 |
|---|---:|---:|---:|
| DeepLab | 0.3994 | 0.3607 | -0.0387 |
| SegFormer | 0.4135 | 0.3787 | -0.0348 |

微调后的 DACL 逐类 IoU：

| 类别 | DeepLab | SegFormer |
|---|---:|---:|
| crack | 0.3852 | 0.3054 |
| spalling | 0.4266 | 0.4362 |
| exposed rebar | 0.3371 | 0.3772 |
| rust | 0.4424 | 0.4955 |
| efflorescence | 0.3373 | 0.3544 |
| wetspot | 0.2356 | 0.3034 |

结论：partial-label masking 避免了把未标注类别直接教成背景，但不能阻止共享解码器发生
间接遗忘。微调 checkpoint 不应直接替代通用六类 checkpoint。若必须产生一个统一模型，
下一轮应采用 DACL rehearsal（混合 DACL 与建筑 batch）或教师蒸馏，而不是增加微调 epoch。

## 9. 最终部署建议

保留并区分两个用途：

```text
通用六类模型：
  runs/segformer_b0_6ml/best.pt

当前建筑 crack/spalling 模型：
  runs/deeplabv3plus_6ml_building/best.pt
```

不要删除或覆盖以下四个 checkpoint：

```text
runs/deeplabv3plus_6ml/best.pt
runs/segformer_b0_6ml/best.pt
runs/deeplabv3plus_6ml_building/best.pt
runs/segformer_b0_6ml_building/best.pt
```

不同域必须使用各自在对应 val 上冻结的阈值。不能拿建筑阈值评估 DACL，也不能拿 DACL
阈值部署建筑数据。

## 10. 关键源码

```text
src/rgb_multilabel/labels.py             六类映射和多边形栅格化
src/rgb_multilabel/masks.py              bit-packed 标签缓存
src/rgb_multilabel/data.py               DACL Dataset 与增强
src/rgb_multilabel/building_data.py      建筑 partial-label Dataset
src/rgb_multilabel/losses.py             masked BCEWithLogits + Dice
src/rgb_multilabel/metrics.py             多标签指标、阈值搜索、relaxed F1
src/rgb_multilabel/models.py              DeepLabV3+ 和 SegFormer 适配器
src/rgb_multilabel/train.py               DACL 统一训练器
src/rgb_multilabel/finetune.py            建筑保守微调器
src/rgb_multilabel/evaluate.py            DACL 评估、速度、显存、可视化
src/rgb_multilabel/evaluate_building.py   建筑 val 校准与 test 评估
```

## 11. 环境

```text
GPU: NVIDIA GeForce RTX 5070 Ti Laptop, 12 GB
Python: 3.12.13（使用 ../../DeepLabs+/.venv）
PyTorch: 2.13.0+cu130
CUDA available: True
输入尺寸: 512x512
```

系统 Python 3.14 不用于该工程。运行命令一般采用：

```bash
cd /home/lucas19980101/projects/InfraDetect/RGB/6ML
PYTHONPATH=src ../../DeepLabs+/.venv/bin/python -m <module>
```

## 12. ChatGPT 接手时的建议提示词

可将本文件上传后发送：

```text
请先完整阅读 PROJECT_HANDOFF_FOR_CHATGPT.md，把它作为本项目的事实来源。
不要假设能够访问此前 Codex 对话。保持现有数据划分和评估协议，不覆盖四个 best.pt。
如果建议继续训练，请明确说明会使用哪个数据域、哪套阈值，以及如何防止六类遗忘。
```
