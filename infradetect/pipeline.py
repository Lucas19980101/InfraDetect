"""单场景二维分析总入口：按顺序生成 IR、RGB、融合与跨模态证据。

输入是场景 manifest，而不是一个任意 RGB/IR 文件夹。标准图像契约是
四通道 NPY：[B, G, R, Thermal]；旧 TBBR 五通道仍可读取，但额外通道
不会自动进入模型。manifest 中还要指出模型、
配准矩阵及可选有效区域。相对路径按交付包根目录解释。

这里负责前置校验、九阶段调度和落盘记录；具体模型推理由 inference/ 执行。
check_only 只做校验，不创建推理结果。输出包含每阶段证据与 status.json，
因此失败后可以知道停在哪一步，但不能把完成状态理解为模型准确率。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def project_root() -> Path:
    """找到交付包路径基准目录；相对清单路径都以此为准。"""
    parent = Path(__file__).resolve().parent.parent
    return parent.parent if parent.name == "IR" else parent


def project_path(root: Path, value: str) -> Path:
    """把 manifest 的相对路径变成绝对路径；绝对路径保持原意。"""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _artifact(root: Path, value: str, label: str) -> Path:
    """解析并检查必需输入文件；缺文件时带着角色名报错。"""
    path = project_path(root, value)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def inspect(manifest: Path, root: Path) -> tuple[dict, dict]:
    """推理前完整校验场景并返回原始清单与可执行的 preflight 摘要。

    验证至少四通道数组、额外通道声明、场景 ID、模型权重/配置、RGB→IR 单应矩阵、
    有效像素掩码及 RGB 裁块/ROI 设置。只在用户明确接受配准时启用
    RGB 证据；不接受配准时仍可保留 IR 分支，但不允许假装 RGB 已对齐。
    此函数不加载模型、不缩放图像，也不写结果。
    """
    # 1. 读清单与场景身份：后续输出目录、报告和三维索引都依赖稳定的 scene_id。
    definition = json.loads(manifest.read_text(encoding="utf-8"))
    if definition.get("schema_version") != 1:
        raise ValueError("Expected pipeline schema_version=1")
    scene = definition.get("scene", {})
    scene_id = scene.get("id", "")
    if not isinstance(scene_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scene_id):
        raise ValueError("Scene id must be a safe name")
    if not scene.get("source_dataset"):
        raise ValueError("Scene requires source_dataset")
    # 2. 核对原生尺寸的四通道图像。这里只 mmap 查看形状，不做 resize。
    source = _artifact(root, scene["image"], "scene image")
    array = np.load(source, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[2] < 4:
        raise ValueError("Scene image must have at least [B,G,R,Thermal] channels")
    shape = tuple(map(int, array.shape[:2]))
    channels = int(array.shape[2])
    # 额外通道必须逐个声明；兼容旧 TBBR 的第五通道不等于启用高度模型。
    extras = scene.get("extra_channels")
    if extras is None or (channels == 5 and extras == []):
        # Preserve existing TBBR stacks, but require names for any new layout.
        extras = ["height_legacy_unused"] if channels == 5 else []
        if channels > 5:
            raise ValueError("Images with >5 channels require scene.extra_channels")
    if (not isinstance(extras, list) or len(extras) != channels - 4
            or any(not isinstance(name, str) or not name.strip() for name in extras)
            or len(set(extras)) != len(extras)):
        raise ValueError("scene.extra_channels must name each channel after [B,G,R,Thermal]")
    metadata = scene.get("capture_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("scene.capture_metadata must be a JSON object")
    # 3. 配准是 RGB 证据进入 IR 坐标的前提；矩阵必须经用户接受且可逆。
    registration = definition.get("registration", {})
    accepted = registration.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("registration.accepted must be true or false")
    matrix = None
    if accepted:
        from scipy.io import loadmat

        matrix = _artifact(root, registration["rgb_to_thermal"], "registration matrix")
        if matrix.suffix.lower() == ".json":
            transform = np.asarray(json.loads(matrix.read_text(encoding="utf-8")), np.float64)
        elif matrix.suffix.lower() == ".npy":
            transform = np.asarray(np.load(matrix, allow_pickle=False), np.float64)
        elif matrix.suffix.lower() == ".mat":
            transform = np.asarray(loadmat(matrix)["H"], np.float64)
        else:
            raise ValueError("Registration matrix must be JSON, NPY, or MAT")
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("Invalid registration matrix")
    # 无效像素只能被排除，不能被当作模型预测的阴性区域。
    if scene.get("valid_mask"):
        valid = cv2.imread(str(_artifact(root, scene["valid_mask"], "valid mask")), cv2.IMREAD_UNCHANGED)
        if valid is None or valid.ndim != 2 or valid.shape != shape:
            raise ValueError("Valid mask must match the scene image")

    # 4. 按角色核对权重和配置。缺 RGB 配准时不加载需要 RGB 对应的模型。
    models = definition.get("models", {})
    required = ["primary_thermal", "diagnostic_thermal"]
    if accepted:
        required.append("components")
    paths = {}
    for role in required + [key for key in ("air_conditioner", "rgb_defect") if key in models]:
        entry = models.get(role)
        if not isinstance(entry, dict) or not entry.get("checkpoint"):
            raise ValueError(f"Missing model role: {role}")
        paths[role] = {"checkpoint": _artifact(root, entry["checkpoint"], f"{role} checkpoint")}
        if role != "rgb_defect":
            if not entry.get("config"):
                raise ValueError(f"Missing model config: {role}")
            paths[role]["config"] = _artifact(root, entry["config"], f"{role} config")
    # RGB 六类分支单独检查滑窗设置和可选 ROI；ROI 不改变原图尺寸。
    if "rgb_defect" in paths:
        paths["rgb_defect"]["source"] = project_path(root, models["rgb_defect"].get("source", "infradetect/inference"))
        if not paths["rgb_defect"]["source"].is_dir():
            raise FileNotFoundError(paths["rgb_defect"]["source"])
        defect_options = models["rgb_defect"]
        patch_size = int(defect_options.get("patch_size", 512))
        stride = int(defect_options.get("stride", 384))
        if patch_size < 1 or not 0 < stride <= patch_size:
            raise ValueError("RGB defect patch_size/stride must satisfy 0 < stride <= patch_size")
        if defect_options.get("roi_mask"):
            roi_path = _artifact(root, defect_options["roi_mask"], "RGB facade/wall ROI")
            roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
            if roi is None or roi.shape != shape:
                raise ValueError("RGB facade/wall ROI must match native image shape")
            paths["rgb_defect"]["roi_mask"] = roi_path
    if not accepted and any(key in models for key in ("components", "air_conditioner", "rgb_defect")):
        raise ValueError("RGB evidence requires accepted registration")

    # 5. 融合参数只是候选筛选与上下文提示规则，并非准确率或物理阈值。
    settings = definition.get("fusion", {})
    threshold = float(settings.get("thermal_threshold", 0.5))
    interference = float(settings.get("interference_threshold", 0.78))
    edge = int(settings.get("edge_radius_ir_pixels", 8))
    if not (0 <= threshold <= 1 and 0 <= interference <= 1 and edge >= 0):
        raise ValueError("Invalid fusion settings")
    # preflight 摘要写入运行目录，便于复现“当时到底用了哪些输入”。
    report = {
        "scene_id": scene_id, "source_dataset": scene["source_dataset"],
        "image": str(source), "shape": list(shape),
        "image_channels": channels, "extra_channels": extras,
        "capture_metadata": metadata,
        "registration_accepted": accepted,
        "rgb_to_thermal": str(matrix) if matrix else None,
        "models": {role: {name: str(path) for name, path in files.items()}
                   for role, files in paths.items()},
        "fusion": {"thermal_threshold": threshold,
                   "interference_threshold": interference,
                   "edge_radius_ir_pixels": edge},
    }
    # PNG 配对适配器不产生真正的测温数据；须保留来源与标签语义警告。
    if scene.get("source_format") == "aligned_png_pair_adapter":
        if channels == 4 and scene.get("height_channel") not in (None, "not_present"):
            raise ValueError("Four-channel facade input must not claim a height channel")
        if (channels == 5 and scene.get("height_channel") != "zero_placeholder_not_measured"):
            raise ValueError("Legacy facade adapter fifth channel needs explicit placeholder provenance")
        if (scene.get("thermal_signal") != "8bit_grayscale_not_radiometric" or
                scene.get("label_semantics") != "generic_binary_facade_defect_not_thermal_anomaly"):
            raise ValueError("Facade adapter provenance is incomplete or misleading")
        provenance_path = _artifact(root, scene["input_provenance"], "facade input provenance")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("shape") != list(shape) or provenance.get("label_semantics") != scene["label_semantics"]:
            raise ValueError("Facade adapter provenance and stack disagree")
        report["input_provenance"] = str(provenance_path)
        report["source_format"] = scene["source_format"]
        report["height_channel"] = scene.get("height_channel", "not_present")
        report["thermal_signal"] = scene["thermal_signal"]
        report["label_semantics"] = scene["label_semantics"]
        report["registration_basis"] = registration.get("basis")
    return definition, report


def _compare(primary: Path, diagnostic: Path, threshold: float,
             valid_mask: Path | None, image: Path, output: Path) -> dict:
    """比较两条热异常分支的逐像素阈值结果，生成分歧诊断图。

    编码 0=都未报、1=主分支独有、2=IR-only 分支独有、3=两者都报；
    有效区域之外不会计入统计。彩色 overlay 只帮助人工定位模型分歧，
    不是缺陷真值，也不是 RGB 与 IR 两种独立模态的确认。
    """
    # 两个输入都必须是同一 IR 像素网格上的 H×W 概率图。
    first = np.load(primary, mmap_mode="r", allow_pickle=False)
    second = np.load(diagnostic, mmap_mode="r", allow_pickle=False)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Primary and diagnostic probability shapes differ")
    valid = np.ones(first.shape, bool)
    if valid_mask is not None:
        mask = cv2.imread(str(valid_mask), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != first.shape:
            raise ValueError("Diagnostic valid-mask shape mismatch")
        valid = mask > 0
    # 位编码 1/2/3 分别表示主分支独有、IR-only 独有、两者同时提示。
    first_positive = (first >= threshold) & valid
    second_positive = (second >= threshold) & valid
    disagreement = first_positive.astype(np.uint8) + 2 * second_positive.astype(np.uint8)
    if not cv2.imwrite(str(output / "diagnostic_disagreement.png"), disagreement):
        raise OSError("Could not write diagnostic disagreement map")
    # 叠加图只方便肉眼找分歧；底图来自原始第 4 通道，不覆盖概率文件。
    source = np.load(image, mmap_mode="r", allow_pickle=False)
    preview = cv2.cvtColor(np.asarray(source[..., 3], dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    for value, color in ((1, (0, 150, 255)), (2, (255, 150, 0)), (3, (40, 210, 40))):
        selected = disagreement == value
        preview[selected] = (0.45 * preview[selected] + 0.55 * np.asarray(color)).astype(np.uint8)
    if not cv2.imwrite(str(output / "diagnostic_disagreement_overlay.png"), preview):
        raise OSError("Could not write diagnostic disagreement overlay")
    return {"valid_pixels": int(valid.sum()),
            "primary_positive_pixels": int(first_positive.sum()),
            "diagnostic_positive_pixels": int(second_positive.sum()),
            "both_positive_pixels": int((first_positive & second_positive).sum()),
            "primary_only_pixels": int((first_positive & ~second_positive).sum()),
            "diagnostic_only_pixels": int((second_positive & ~first_positive).sum()),
            "meaning": "model disagreement for review, not a defect label",
            "map_legend": {"0": "neither", "1": "primary_only",
                           "2": "diagnostic_only", "3": "both"}}


def run(manifest: Path, output: Path, root: Path | None = None,
        check_only: bool = False) -> dict:
    """执行九阶段二维流水线，或在 check_only 模式仅返回预检结果。

    正式运行要求输出目录为空，保存 manifest 快照和 preflight；先生成
    主热异常与独立 IR-only 证据，再在配准有效时生成 RGB 证据，随后做
    融合、对应和综合评判。status.json 记录当前阶段，异常会保留已有
    中间结果供排查；不要用同一路径覆盖旧实验。
    """
    # 阶段 0：只读 preflight。--check 到这里结束，不创建模型输出。
    root = (root or project_root()).resolve()
    definition, preflight = inspect(manifest.resolve(), root)
    if check_only:
        return preflight
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to preserve previous results: {output}")
    # 各模块按职责放在同一 infradetect 包中；不再依赖动态修改 sys.path。
    from .evidence.export_rgb import run as export_rgb
    from .evidence.export_thermal import run as export_thermal
    from .evidence.fusion import run as fuse
    from .evidence.cross_modal import run as compare_modalities

    # 保存输入快照，避免日后原 manifest 被改后误读旧运行结果。
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest_snapshot.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    scene = definition["scene"]
    image = project_path(root, scene["image"])
    models = preflight["models"]
    status = {"stage": "primary_thermal", "complete": False}
    try:
        # 阶段 1～2：两条热分支各自产生原 IR 坐标概率；IR-only 不被 RGB 替代。
        print("[1/9] primary thermal model", flush=True)
        primary = models["primary_thermal"]
        primary_summary = export_thermal(image, Path(primary["config"]),
                                         Path(primary["checkpoint"]), output / "thermal_primary")
        status["stage"] = "diagnostic_thermal"
        print("[2/9] diagnostic thermal model", flush=True)
        diagnostic = models["diagnostic_thermal"]
        diagnostic_summary = export_thermal(image, Path(diagnostic["config"]),
                                            Path(diagnostic["checkpoint"]), output / "thermal_diagnostic")
        # 阶段 3：只有配准已接受时，才生成 RGB 构件、空调和可见缺陷证据。
        rgb_summary = None
        if preflight["registration_accepted"]:
            status["stage"] = "rgb_evidence"
            print("[3/9] RGB evidence", flush=True)
            components = models["components"]
            ac = models.get("air_conditioner")
            defect = models.get("rgb_defect")
            rgb_summary = export_rgb(
                image, output / "rgb", Path(components["config"]), Path(components["checkpoint"]),
                Path(ac["config"]) if ac else None,
                Path(ac["checkpoint"]) if ac else None,
                Path(defect["checkpoint"]) if defect else None,
                Path(defect["source"]) if defect else None,
                rgb_patch_size=int(definition["models"]["rgb_defect"].get("patch_size", 512)) if defect else 512,
                rgb_stride=int(definition["models"]["rgb_defect"].get("stride", 384)) if defect else 384,
                rgb_roi_mask=Path(defect["roi_mask"]) if defect and "roi_mask" in defect else None)
        # 阶段 4：写清每个输入证据的路径，让融合模块在同一坐标系中处理。
        status["stage"] = "fusion"
        print("[4/9] primary evidence fusion", flush=True)
        fused_scene = {
            "id": scene["id"], "source_dataset": scene["source_dataset"],
            "thermal_model": primary["checkpoint"],
            "thermal_input_mode": primary_summary["input_mode"],
            "thermal_probability": str((output / "thermal_primary" / "thermal_probability.npy").resolve()),
            "thermal_preview": str(image),
            "registration_accepted": preflight["registration_accepted"],
        }
        if scene.get("valid_mask"):
            fused_scene["valid_mask"] = str(project_path(root, scene["valid_mask"]))
        if preflight["registration_accepted"]:
            fused_scene["rgb_to_thermal"] = preflight["rgb_to_thermal"]
            fused_scene["component_ids_rgb"] = str((output / "rgb" / "component_ids_rgb.png").resolve())
            fused_scene["wall_probability_rgb"] = str((output / "rgb" / "wall_probability_rgb.npy").resolve())
            if "air_conditioner" in models:
                fused_scene["interference_probability_rgb"] = str((output / "rgb" / "interference_probability_rgb.npy").resolve())
            if "rgb_defect" in models:
                fused_scene["rgb_semantic_probabilities"] = str((output / "rgb" / "rgb_semantic_probabilities.npz").resolve())
        fusion_manifest = {"schema_version": 1, "fusion": preflight["fusion"], "scenes": [fused_scene]}
        fusion_manifest_path = output / "fusion_manifest.json"
        fusion_manifest_path.write_text(json.dumps(fusion_manifest, indent=2) + "\n", encoding="utf-8")
        fused = fuse(fusion_manifest_path, output / "fusion")[0]
        # 阶段 5：换成 IR-only 概率，再跑同一融合规则，供分支对照。
        status["stage"] = "diagnostic_fusion"
        print("[5/9] thermal-only evidence fusion", flush=True)
        diagnostic_scene = dict(fused_scene)
        diagnostic_scene["thermal_model"] = diagnostic["checkpoint"]
        diagnostic_scene["thermal_input_mode"] = diagnostic_summary["input_mode"]
        diagnostic_scene["thermal_probability"] = str(
            (output / "thermal_diagnostic" / "thermal_probability.npy").resolve())
        diagnostic_manifest = {**fusion_manifest, "scenes": [diagnostic_scene]}
        (output / "fusion_diagnostic_manifest.json").write_text(
            json.dumps(diagnostic_manifest, indent=2) + "\n", encoding="utf-8")
        diagnostic_fused = fuse(output / "fusion_diagnostic_manifest.json",
                                output / "fusion_diagnostic")[0]
        # 阶段 6～7：RGB/热图只做精确或邻近的空间对应，不自动认定同一病因。
        cross_modal = None
        cross_modal_diagnostic = None
        if "rgb_defect" in models:
            print("[6/9] RGB and primary-model correspondence", flush=True)
            status["stage"] = "cross_modal"
            cross_modal = compare_modalities(output / "fusion" / scene["id"],
                                             output / "rgb" / "summary.json",
                                             output / "cross_modal",
                                             preflight["fusion"]["edge_radius_ir_pixels"])
            print("[7/9] RGB and thermal-only correspondence", flush=True)
            cross_modal_diagnostic = compare_modalities(
                output / "fusion_diagnostic" / scene["id"],
                output / "rgb" / "summary.json", output / "cross_modal_diagnostic",
                preflight["fusion"]["edge_radius_ir_pixels"])
        # 阶段 8：汇集证据及两热分支分歧；报告不是最终工程诊断。
        status["stage"] = "report"
        print("[8/9] report and model disagreement", flush=True)
        comparison = _compare(output / "thermal_primary" / "thermal_probability.npy",
                              output / "thermal_diagnostic" / "thermal_probability.npy",
                              preflight["fusion"]["thermal_threshold"],
                              project_path(root, scene["valid_mask"]) if scene.get("valid_mask") else None,
                              image, output)
        report = {"schema_version": 1, "scene": preflight,
                  "primary_thermal": primary_summary,
                  "diagnostic_thermal": diagnostic_summary,
                  "rgb": rgb_summary, "fusion": fused,
                  "fusion_diagnostic": diagnostic_fused,
                  "cross_modal": cross_modal,
                  "cross_modal_diagnostic": cross_modal_diagnostic,
                  "diagnostic_comparison": comparison,
                  "fusion_output": str((output / "fusion" / scene["id"]).resolve()),
                  "three_dimensional_projection": "not_run"}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # 阶段 9：形成待人工复核的区域级优先级；不抹掉任何 IR 候选。
        if preflight["registration_accepted"] and cross_modal is not None:
            from .evidence.composite_assessment import assess

            status["stage"] = "composite_assessment"
            print("[9/9] conservative composite assessment", flush=True)
            assessment = assess(output, output / "composite_assessment")
            report["composite_assessment"] = {
                "summary": str((output / "composite_assessment" / "summary.json").resolve()),
                "candidate_regions": assessment["candidate_regions"],
                "engineering_diagnosis": assessment["engineering_diagnosis"],
            }
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        status = {"stage": "complete", "complete": True}
        return report
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # 成功或失败都留下最后阶段，便于定位中断位置并保护已生成的证据。
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """解析命令行；--check 只校验，其余参数交给 run 执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate inputs without running models")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output, args.project_root, args.check), indent=2))
