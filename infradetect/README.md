# `infradetect/`：场景分析与人工复核代码导读

这个目录是交付包里负责“把模型结果串起来”的代码，不是训练代码。建议按下面顺序读。每个文件开头说明输入、处理、输出，复杂函数内也按小阶段标了注释。

1. `examples/scene.example.json`：描述一次分析需要哪些图像、模型和配准信息。JSON 不支持注释，字段解释见 `examples/README.md`。
2. `__main__.py` → `pipeline.py`：命令行先校验清单，再调度二维证据生成。
3. `inference/`：加载热模型、构件/空调模型和六类 RGB 模型，在原像素网格上滑窗预测。
4. `evidence/`：导出概率图与覆盖区，比较已配准的 RGB/IR 证据，并生成复核提示。
5. `geometry/`：若有 COLMAP 模型和深度图，投影候选并核对多照片中的可见性。
6. `tbbr_review.py` + `tbbr_review.html`：把点云、照片、候选与人工判断显示在本机网页中。

`inference/` 的文件名按职责区分：`rgb_predict.py` 负责加载 RGB 权重并在
原图裁片上计算概率；`thermal_context_architectures.py` 定义热异常及
RGB 构件/空调网络；`rgb_defect_architectures.py` 只定义六类 RGB 缺陷
网络。两个 `architectures` 文件都不读取图像、不执行滑窗，也不保存结果。

## 一张图如何走完二维流程

`pipeline.inspect()` 读入场景 JSON，检查图像至少有 `[B,G,R,Thermal]` 四通道、模型文件存在、有效区尺寸正确、RGB→IR 配准矩阵可用。这里只验证文件和数值，**不能证明配准在视觉上准确**。`--check` 到此为止。

`pipeline.run()` 的正式运行步骤：

1. 主热模型生成热异常概率；独立 IR-only 模型另生成一张概率图。
2. 配准经接受时，RGB 模型生成墙面/构件、空调设备及可见缺陷证据。没有合格配准就不声称 RGB 支持 IR。
3. 两条热分支分别与上下文融合。融合结果只调整候选解释，不把干扰线索等同于“异常已被否定”。
4. `cross_modal` 比较 RGB 与热异常的像素共址或邻近关系。
5. `composite_assessment` 把热候选连成区域，计算可追溯的证据计数和人工复核顺序。
6. 写 `report.json`、各阶段文件和 `status.json`。`complete=true` 仅说明流程跑完，不代表模型准确。

二维图始终在原 IR 像素坐标中对照；RGB 缺陷推理可用原图滑窗，不要求先把整张图缩成 512×512。三维核验是后续独立步骤，**不能从单张二维图凭空得到可靠 3D**。

## 每个模块的“输入 → 处理 → 输出”

- `pipeline.py`：场景 manifest → 输入核验与阶段调度 → 每阶段证据、`report.json`、`status.json`。`_compare()` 只可视化主热与 IR-only 分歧。
- `evidence/cross_modal.py`：IR 网格上的热概率、RGB 六类概率、有效区和各类阈值 → 计算精确/半径内对应 → 状态 PNG、类别位图、区域 JSON。`255` 是未评估，不能当阴性。
- `evidence/composite_assessment.py`：两热分支 mask 与墙/设备/RGB 上下文 → 热候选并集、连通区域、证据计数 → 轮廓图与复核摘要。`review_priority` 未定标，不是缺陷概率。
- `geometry/multiview.py`：投影候选 PLY、COLMAP 相机/深度、逐图二维证据 → 可见性和视角独立性检查 → 点级、组级 JSON 与原始观测 NPZ。遮挡点不计为漏检。
- `tbbr_review.py`：TBBR、BFDD、模型审计或几何基线中的一种数据 → 适配为统一的点云/照片 payload，并运行本机 HTTP 服务 → 独立的 `reviews.json`、`group_reviews.json`。它不修改原标注。
- `tbbr_review.html`：服务端 payload → 点云和原图/叠加层的交互展示 → 用户显式点击才向本机 API 提交判断。黄圈和 mask 可单独关闭；缩放只影响显示。

## 容易误读的几件事

- 热异常候选、RGB 可见缺陷、空调/材质干扰、几何偏离是不同证据。任意两种共址都不自动等于同一病因。
- BFDD 彩色点是原标签通过稀疏轨迹投到三维，稠密点云只是几何背景；不是“稠密点逐点真值”。
- 几何偏离可能来自窗、栏杆、另一面墙或重建噪声；需要照片与人工判断。
- 相邻点的分组是为了减少复核工作量，不证明它们属于一处物理缺陷。
- 文件中的 `valid`、配准覆盖及 ROI 外区域都是“未评估”；不要把它们计作模型判正常。
