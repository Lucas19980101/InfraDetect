#!/usr/bin/env python3
"""Convert a FAST-LIVO2 binary/ascii PCD crop to the Real3D-AD NPZ contract."""

import argparse
from pathlib import Path

import numpy as np


PCD_TYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4",
}


def read_pcd(path: Path):
    header = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: incomplete PCD header")
            decoded = line.decode("ascii").strip()
            if decoded and not decoded.startswith("#"):
                key, *values = decoded.split()
                header[key.upper()] = values
            if decoded.upper().startswith("DATA "):
                break
        fields = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if any(count != 1 for count in counts):
            raise ValueError("COUNT > 1 fields are not supported")
        dtype = np.dtype([(name, PCD_TYPES[(kind, size)]) for name, kind, size in zip(fields, types, sizes)])
        mode = header["DATA"][0].lower()
        if mode == "binary":
            cloud = np.fromfile(handle, dtype=dtype, count=int(header["POINTS"][0]))
        elif mode == "ascii":
            values = np.loadtxt(handle)
            cloud = np.empty(len(values), dtype=dtype)
            for index, name in enumerate(fields):
                cloud[name] = values[:, index]
        else:
            raise ValueError(f"Unsupported PCD DATA mode: {mode}")
    xyz = np.column_stack([cloud[name] for name in ("x", "y", "z")]).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], np.flatnonzero(finite)


def densest_block(points: np.ndarray, size: float):
    cells = np.floor(points / size).astype(np.int64)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    selected = int(np.argmax(counts))
    return (unique[selected].astype(np.float64) + 0.5) * size, inverse == selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", type=float, default=10.0, help="axis-aligned crop side in metres")
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()
    if args.size <= 0 or args.max_points <= 0:
        parser.error("--size and --max-points must be positive")

    points, source_indices = read_pcd(args.input)
    if args.center is None:
        center, keep = densest_block(points, args.size)
    else:
        center = np.asarray(args.center, dtype=np.float64)
        keep = np.all(np.abs(points - center) <= args.size / 2, axis=1)
    points, source_indices = points[keep], source_indices[keep]
    if not len(points):
        raise ValueError("crop contains no points")
    if len(points) > args.max_points:
        choice = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
        choice.sort()
        points, source_indices = points[choice], source_indices[choice]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, points=points, mask=np.zeros(len(points), dtype=np.uint8),
                        source_indices=source_indices, crop_center=center, crop_size=args.size,
                        source_pcd=str(args.input))
    print(f"Wrote {len(points)} points around {center.tolist()} to {args.output}")


if __name__ == "__main__":
    main()
