"""Create deterministic group-safe train/val/test splits for prepared component data."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


def family(source: str) -> str:
    if source.startswith("cmp_"):
        return "cmp"
    if source.startswith("hznu"):
        return "hznu"
    return source


def stable_order(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def assign(rows: list[dict], seed: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["source"] == "3dom" or row["split"] == "exclude":
            continue
        groups[(family(row["source"]), row["name"])].append(row)
    by_family: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for (source_family, name), members in groups.items():
        by_family[source_family].append((name, members))
    for source_family, items in by_family.items():
        items.sort(key=lambda item: stable_order(f"{source_family}/{item[0]}", seed))
        total = len(items)
        val_count = round(total * 0.15)
        test_count = round(total * 0.15)
        for index, (_, members) in enumerate(items):
            split = "val" if index < val_count else "test" if index < val_count + test_count else "train"
            for row in members:
                row["split"] = split
                row["group"] = f"{source_family}/{row['name']}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/components_3class/manifest_with_splits.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/components_3class/manifest_grouped_splits.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = assign(list(reader), args.seed)
    if fieldnames is None:
        raise ValueError(f"Empty manifest: {args.input}")
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    counts = Counter((row["source"], row["split"]) for row in rows)
    for (source, split), count in sorted(counts.items()):
        print(f"{source}/{split}: {count}")


if __name__ == "__main__":
    main()
