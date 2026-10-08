from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path

from .data import component_pairs


def capture_time(name: str) -> datetime:
    return datetime.strptime(name.split("_")[1], "%Y%m%d%H%M%S")


def bfdd_rows(root: Path) -> list[dict[str, str]]:
    train = set((root / "train.txt").read_text(encoding="utf-8").split())
    test = set((root / "test.txt").read_text(encoding="utf-8").split())
    images = {p.stem for p in (root / "IR").glob("*.png")}
    if train & test or train | test != images:
        raise ValueError("BFDD train/test lists overlap or do not cover IR images")
    names = sorted(images, key=capture_time)
    groups: list[list[str]] = []
    for name in names:
        if not groups or (capture_time(name) - capture_time(groups[-1][-1])).total_seconds() > 30:
            groups.append([])
        groups[-1].append(name)
    eligible = [group for group in groups if set(group).isdisjoint(test)]
    eligible.sort(key=lambda group: hashlib.sha256(group[0].encode()).hexdigest())
    target = round(len(train) * 0.15)
    val: set[str] = set()
    for group in eligible:
        if len(val) >= target:
            break
        val.update(group)
    if not val or val == train:
        raise ValueError("Could not create a train-only validation set")
    return [{"name": name, "split": "test" if name in test else "val" if name in val else "train",
             "group": str(next(i for i, group in enumerate(groups) if name in group))} for name in names]


def bfdd_specialized_rows(root: Path) -> list[dict[str, str]]:
    """Leakage-resistant split for the 838-image local BFDD collection.

    A new session starts after a gap longer than five minutes. Reviewed sessions
    are kept date-disjoint: June 24-25 train, June 28 validation, and June 26
    test. This intentionally does not preserve the leaky provider split. June 26
    sessions 7 and 8 remain together because visual review shows the same facade.
    """
    images = {p.stem for p in (root / "IR").glob("*.png")}
    names = sorted(images, key=capture_time)
    groups: list[list[str]] = []
    for name in names:
        if not groups or (capture_time(name) - capture_time(groups[-1][-1])).total_seconds() > 300:
            groups.append([])
        groups[-1].append(name)
    expected_sizes = [12, 82, 219, 111, 56, 14, 76, 117, 107, 24, 20]
    if [len(group) for group in groups] != expected_sizes:
        raise ValueError("BFDD specialized split expects the reviewed 838-image, 11-session collection")
    rows = []
    for group_id, group in enumerate(groups):
        split = "train" if group_id <= 6 else "test" if group_id <= 8 else "val"
        rows.extend({"name": name, "split": split, "group": f"session_{group_id:02d}"}
                    for name in group)
    return rows


def component_group(name: str) -> str:
    # 3DOM's five scene families are not explicitly identified per image.
    # These conservative filename groups keep closely named views together.
    if name.startswith("Navona_"):
        return "Navona"
    if name.startswith("IMG_") or name.startswith("image"):
        return "IMG_family"
    if name.startswith("_DSC"):
        return name[:5]
    if name.startswith("DSC_"):
        return "DSC"
    return name.split("_")[0]


def component_rows(root: Path) -> list[dict[str, str]]:
    pairs = component_pairs(root)
    groups: dict[str, list[str]] = {}
    for image, _ in pairs:
        groups.setdefault(component_group(image.stem), []).append(image.stem)
    if len(groups) < 3:
        raise ValueError(f"Only {len(groups)} component groups; review split manually")
    target = round(len(pairs) * 0.15)
    eligible = sorted((group for group in groups if len(groups[group]) <= target),
                      key=lambda group: (-len(groups[group]), hashlib.sha256(group.encode()).hexdigest()))
    held_out: dict[str, str] = {}
    totals = {"val": 0, "test": 0}
    for group in eligible:
        if min(totals.values()) >= target:
            break
        split = min(totals, key=lambda candidate: (totals[candidate], candidate))
        held_out[group] = split
        totals[split] += len(groups[group])
    if min(totals.values()) == 0:
        raise ValueError("Could not form both component validation and test groups")
    # Hold out entire filename families. The manifest is reviewable before training.
    return [{"name": name, "split": held_out.get(group, "train"),
             "group": group} for group, names in sorted(groups.items()) for name in sorted(names)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("ir", "ir_specialized", "components"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = (bfdd_rows(args.root) if args.task == "ir" else
            bfdd_specialized_rows(args.root) if args.task == "ir_specialized" else
            component_rows(args.root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "split", "group"))
        writer.writeheader(); writer.writerows(rows)
    from collections import Counter
    print(f"{args.output}: {dict(Counter(row['split'] for row in rows))}")


if __name__ == "__main__":
    main()
