# 三类构件数据整理（未训练）

转换脚本：`python -m seg_tasks.prepare_components_3class`。原始文件只读；它优先读取已解压的 `Hznu Facade Dataset/original/all_adjust_zhedang/`，若该目录不存在才读取原始 ZIP。输出的 `manifest.csv` 逐图记录来源、原图、原标签、转换后单通道 PNG 路径以及四种 ID 的像素数。`audit.json` 记录类别映射和异常项。`previews/` 中的 JPG 是原图叠图，只用于人工检查。`images/hznu/` 是 Hznu 照片副本；带 EXIF 方向的副本会先转正并清除方向标记，JSON 多边形随后在这个转正坐标系中重画。每次运行都会重建 Hznu mask，避免错误坐标系产生的旧 mask 被缓存复用。

运行 `python -m seg_tasks.finalize_components_3class` 后，`manifest_with_splits.csv` 和 `validation.json` 会核对所有原图/标签尺寸与类别值，并保留 3DOM 原有 train/val/test；CMP、eTRIMS-8、Hznu 暂全部设为 train，避免同源近重复图像进入验证集。这个划分只是训练准备，不代表已经验证了 BFDD 目标域。

人工审查中确认有问题的样本记录在 `exclusions.csv`。`finalize_components_3class` 会把这些行的 `split` 设为 `exclude`；原图、原标签和转换 mask 仍保留，待人工修正后可复审。训练读取器必须只读取 `split=train`，不能把 `exclude` 当训练图。

Hznu 原始 JPEG 另有 EXIF 方向风险。运行 `python -m seg_tasks.audit_hznu_orientation` 会生成 `hznu_orientation_review.csv`；发现 77 张带 90° 方向，其中 70 张 JSON 缺少尺寸元数据，7 张尺寸与 JPEG 像素宽高对调。`finalize_components_3class` 默认隔离这些样本；人工确认可用的名单记录在 `hznu_orientation_accepted.csv`，明确有问题的样本仍记录在 `exclusions.csv`，且人工排除优先。结合后续全量 `original` 叠图复核，当前 Hznu original 为 484 train / 140 exclude；范围式问题报告不会覆盖此前已经确认可用的样本。

统一 ID：`0=building_surface`、`1=opening`、`2=other`、`255=ignore`。

- 3DOM：facade→0；window/door→1；sky/obstacle→2；原始 0 边界→255。
- CMP base/extended：facade/cornice/sill/deco/molding/pillar→0；window/door/blind→1；background/balcony/shop/未标注→255。CMP 不提供可靠的 `other` 标注，不用它的 `background` 冒充天空。
- eTRIMS-8：building→0；window/door→1；car/pavement/road/sky/vegetation→2；索引 0→255。只用 8 类版本，不重复加入同一照片的 4 类版本。
- Hznu original：building→0；window/door→1；sky/tree/car→2；多边形外及异常类别→255。多边形按 building→other→opening 的优先级栅格化；透视矫正版未加入，避免与原始照片重复。

转换完成后应至少检查每来源 6 张 `previews/`。尤其复核 CMP 装饰与窗台是否属于期望保留的建筑表面、eTRIMS 的窗/门是否对应，以及 Hznu 的建筑多边形是否覆盖非墙面。任何复核前都不应启动混合训练。BFDD 现有 `Label` 是缺陷标签，不是构件真值；不能拿来评估本模型。

## 人工抽查叠图

从 `IR/` 运行：

```bash
python -m seg_tasks.review_components_3class --source hznu --count 20
```

输出到 `data/components_3class/review/hznu/`。`contact_sheet.jpg` 是总览，每个 `c_*.jpg` 或 `d_*.jpg` 是左原图、右标签覆盖图。抽样按相机 `c_` 与无人机 `d_` 各半、编号均匀选取，结果固定可复核。蓝色为建筑表面，绿色为窗/门，黄色为其他；保持原图颜色的是 `255=ignore`。重点检查蓝色是否盖住架空入口、玻璃、天空、地面，以及绿色是否漏掉窗门。

如需指定图片：`python -m seg_tasks.review_components_3class --source hznu --names d_0001 d_0109`。同一脚本也支持 `--source cmp_base`、`cmp_extended`、`etrims8` 或 `3dom`。它只写审查图，不会修改原始数据或训练标签。

要生成 Hznu 全部 `c_` 与 `d_` 照片的左右对照图：`python -m seg_tasks.review_components_3class --source hznu --all`。每张输出一个同名 JPG，另按每 20 张分成 `contact_sheet_001.jpg`、`contact_sheet_002.jpg` 等总览图，避免单张总览过高。已有 `exclude` 样本也会显示，供人工复核，但不会因此重新进入训练集。

只复查 77 张 EXIF 方向隔离图：`python -m seg_tasks.review_components_3class --source hznu --exif-only --output data/components_3class/review_exif`。输出在 `review_exif/hznu/`，包括逐张左右对照图及 4 张分页总览。此模式将照片按 EXIF 转正，再直接用原始 JSON 在转正后的坐标系中重画标签；**不使用现有的训练 mask**。部分训练 mask 曾按 JPEG 未转正尺寸生成并截断 JSON 坐标，因此叠图对准也不表示训练数据已修复；确认后须重新生成相应 mask 再纳入训练。

只生成尚未复核的 EXIF 样本：`python -m seg_tasks.review_components_3class --source hznu --exif-pending --output data/components_3class/review_exif_pending`。该模式强制从 `collected_datasets/Hznu Facade Dataset/original/` 中读取原始 JPEG 及其 JSON，不会使用 `IR/data/components_3class/images/` 中的训练副本。当前 EXIF 样本已全部复核，没有待复核项。

生成当前全部排除样本：`python -m seg_tasks.review_components_3class --source hznu --excluded-only --output data/components_3class/review_excluded_current`。同样直接读取 `collected_datasets` 原始 JPEG 和 JSON；当前为 44 张逐图叠图及 3 页总览。

`perspective correction` 版本尚未加入训练 manifest。该版本人工确认不可用的样本单独记录在 `hznu_perspective_exclusions.csv`，避免与 `original` 的排除清单混淆；当前 `c_0310` 因源 JPG 横置且 JSON 仅覆盖旋转后部分区域而排除。

复核 `original` 源图可运行 `python -m seg_tasks.review_hznu_folder --input "../collected_datasets/Hznu Facade Dataset/original/all_adjust_zhedang" --output data/components_3class/review_original_pending --skip-csv data/components_3class/hznu_orientation_accepted.csv data/components_3class/exclusions.csv`。脚本直接读取源 JPG/JSON。全量检查后，用户报告的问题图已全部写入 `exclusions.csv`；未列出的 original 图按本轮复核结果保留使用。`hznu_orientation_accepted.csv` 只记录曾因 EXIF 被隔离后恢复的子集，不是全部可用图片清单。
