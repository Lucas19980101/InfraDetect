# 场景清单示例说明

`scene.example.json` 是填写模板，不是可直接运行的数据。复制后至少替换
`scene.id`、`scene.source_dataset`、`scene.image` 并核验
`registration`。目前统一二维入口要求 `scene.image` 指向
`H×W×4` NPY（通道 `[B,G,R,Thermal]`），不能把两张未对齐照片
的路径直接填入。`models` 中的路径指向随包的五个权重；更换权重时还要
更换匹配的 YAML 配置并重新预检。

`registration.accepted=true` 是使用者声明已核验 RGB→IR 配准，程序只能
检查矩阵形状和基本数值，不会自动验证画面是否真正对齐。模板里的单位
变换仅适用于同像素对齐的 TBBRv2 栈。未核验的新场景不能照搬。
示例矩阵文件与本模板同在 `infradetect/examples/`，不属于任何模型权重；
换场景时应使用该场景经过核验的配准矩阵。
`fusion` 中的阈值是候选生成/复核规则，不是经目标建筑真值标定的确诊阈值。
相对路径按交付包根目录解析，**不是**按本 JSON 所在目录解析。

`scene.extra_channels` 是可选的附加**逐像素**通道名称列表：四通道时填
`[]`；如果有 `H×W×6` 图，且第 5、6 通道依次是 Height、反射率，
可填 `["height", "reflectance"]`。新布局超过五通道时必须声明；
旧 TBBR 五通道在不声明时仍按未使用的 Height 兼容读取。当前推理只用
前四通道，附加名称只用于核验和追溯，不会让模型自动学会利用新传感器。
`scene.capture_metadata` 是可选 JSON 对象，用来记录整张图共享的
GPS、拍摄时间、相机方位等；它会进入运行摘要，但目前不参与推理。

JSON 格式本身不支持注释，因此字段解释放在本文件和
[`ARCHITECTURE.md`](../../ARCHITECTURE.md)；请保持示例文件仍可被严格
JSON 解析。

## 模板逐块注解

`schema_version` 固定为 `1`，用于防止新版/旧版字段被悄悄混读。

`scene` 是这次分析的对象：

- `id`：本次场景的唯一、安全文件名；也用于输出子目录。
- `source_dataset`：图像来源名称，只做追溯，不自动决定标签真值。
- `image`：单张 `H×W×4` 原始尺寸 NPY。前 3 通道按 B、G、R 顺序存 RGB 证据，第 4 通道是 Thermal；若多通道须在 `extra_channels` 逐一声明。
- `extra_channels`：第四通道之后的名称列表。当前附加通道只是记录和校验，不会自动参与已交付模型推理。
- `capture_metadata`：整张图的 GPS、时间、方位等可选元数据对象；当前不会改变模型预测。
- 可选 `valid_mask`：和热图同高同宽的单通道图；0 表示不可评估，非零表示可评估。它不是缺陷标签。

`registration` 说明 RGB 如何对应热图：

- `accepted`：必须由使用者明确填 `true` 或 `false`。矩阵数值检查通过并不代表画面肉眼对齐，所以不要自动设为 true。
- `rgb_to_thermal`：RGB 像素到 Thermal 像素的 3×3 单应矩阵文件；只在 `accepted=true` 时用于跨模态对照。不能拿别的场景矩阵直接套用。

`models` 按功能角色填写：

- `primary_thermal`：RGB+Thermal 主热候选模型，填对应的 `config` 与 `checkpoint`。
- `diagnostic_thermal`：独立 Thermal-only 对照模型，保留不依赖 RGB 的热证据。
- `components`：墙、开口等 RGB 构件上下文；需要可接受的配准。
- `air_conditioner`：空调设备提示，只覆盖其训练语义，不能排除所有热干扰。
- `rgb_defect`：六类 RGB 可见缺陷模型。`source` 保留为兼容字段，
  示例指向 `infradetect/inference`；网络代码已内置在此包，不再从该路径
  动态导入。`patch_size` 和 `stride` 控制原图滑窗，不是整图缩放尺寸。
  可选 `roi_mask` 限制评估范围，ROI 外不能当阴性。

`fusion` 是候选解释规则：

- `thermal_threshold`：把热概率转成候选 mask 的阈值。
- `interference_threshold`：设备/干扰提示所用阈值，不是物理温度门槛。
- `edge_radius_ir_pixels`：RGB 缺陷与热候选“附近”判断的 IR 像素容差；精确重合另算。

运行前可先用 `python -m infradetect --manifest 路径 --output 新目录 --check` 做只读预检；正式运行时去掉 `--check`。输出目录请使用新目录，避免覆盖先前证据。
