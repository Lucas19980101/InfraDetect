# 随包模型索引

这里只收录当前 `infradetect.pipeline` 清单实际调用的五个 **best** 模型；不是
InfraDetect 所有历史实验的罗列。GitHub 版每个目录都有模型卡；从单独提供的
`models_weights.zip` 解压后，每个目录才会有 `checkpoint.pt`。
四个使用独立 YAML 的模型还把 `config.yaml` 放在同一目录。六类 RGB 模型的
推理配置保存在 checkpoint 内，训练参数另见其 `training_config.yaml`。
配置字段说明见 [CONFIGS.md](CONFIGS.md)。模型卡中的
IoU/Dice 是对应数据集上的分割指标，**不是**项目目标建筑的最终系统准确率。

五个模型按用途分组如下：

1. **热图主分支**：[RGB+Thermal](thermal/primary_rgb_thermal/README.md)。
   找热异常候选；评估场景是 TBBR 屋顶热桥，不是空鼓真值。
2. **热图对照分支**：[Thermal-only](thermal/diagnostic_thermal_only/README.md)。
   提供独立的热图证据；与主分支使用同一 TBBR 划分，可作模型对照。
3. **RGB 上下文**：[立面构件](rgb/facade_components/README.md)。
   区分墙面、开口和其他构件；多来源语义标签不是缺陷标签。
4. **RGB 干扰提示**：[空调设备](rgb/air_conditioner/README.md)。
   标记可能的设备热源；HKUST 跨域空调 IoU 很低，不能排除全部干扰。
5. **RGB 可见缺陷**：[六类多标签](rgb/visible_defects/README.md)。
   提供裂缝等可见证据；当前交付的是 DACL 原尺度 class-aware 训练的
   Experiment B。A/B/C70 的训练设置和验证对照详见该模型卡。

各 `checkpoint.pt` 是从原实验 `best.pt` 转出的**推理版**：所有模型权重张量逐一
比对完全相同，保留推理所需 `model/config/epoch/thresholds`，移除 optimizer、
scheduler、scaler。它们不能用于原样恢复训练；原始全量 checkpoint 未改动。
读取 checkpoint 使用 PyTorch 的 `weights_only=True` 受限加载；文件仍须来自可信来源。
传输后可在本目录运行 `sha256sum -c SHA256SUMS` 核对五个推理文件。

示例清单 [`../infradetect/examples/scene.example.json`](../infradetect/examples/scene.example.json)
已指向这些分类目录。标准输入为已配准的四通道 BGR+Thermal 场景栈，
旧 TBBRv2 五通道栈仍可读但 Height 不参与当前推理；新设备/新场景必须
单独核验 RGB→IR 配准，不能默认沿用单位矩阵。模型的数字仅说明现有基准表现，
不能把 RGB 缺陷、热异常或空调分割自动解释为空鼓/污渍等病害确诊。
