# 随包模型配置怎么读

四份 `config.yaml` 分别与其所在目录的 `checkpoint.pt` 配对，属于**配置快照**，
不是开箱即用的训练配方。将配置和权重放在一起，便于查看、复制和替换。
推理程序主要读取 `task`、`data.class_names`、输入模态、归一化、
`data.image_size` / `native_patches`、`model` 网络结构；加载
checkpoint 时还会核对关键字段是否一致。

- `thermal/primary_rgb_thermal/config.yaml`：
  主热异常分支，输入已配准的 RGB+IR，输出二类热异常概率。
- `thermal/diagnostic_thermal_only/config.yaml`：
  IR-only 对照分支，用于保留不依赖 RGB 的热证据。
- `rgb/facade_components/config.yaml`：RGB 墙面/开口/其他构件证据。
- `rgb/air_conditioner/config.yaml`：RGB 空调设备干扰线索。

六类 RGB 可见缺陷模型另有
`rgb/visible_defects/training_config.yaml`，记录当前 B 权重的原始训练参数。
**推理程序不读取这份 YAML**；其结构、类别顺序和阈值由
`rgb/visible_defects/checkpoint.pt` 提供。训练思路与结果见对应模型卡。

`data.root`、`data.manifest`、`include_sources`、`augmentation` 和
`train` 段保留历史训练来源及超参数，交付包**没有**这些数据或训练脚本。
不要为“修复路径”直接修改配置并覆盖原文件；换输入或权重应复制成新配置，
重新核对通道顺序、归一化和模型结构。

示例 `infradetect/examples/tbbr_stacked_rgb_to_thermal_identity.json` 不是模型配置，
而是 RGB→IR 的 3×3 单位矩阵，只能用于确实已同像素对齐的场景栈。
其他相机/场景必须使用人工核验的变换；JSON 不允许 `#` 或 `//` 注释，
详情见 [`../infradetect/examples/README.md`](../infradetect/examples/README.md)。
