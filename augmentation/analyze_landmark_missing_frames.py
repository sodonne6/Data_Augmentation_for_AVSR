from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np


@dataclass
class FileStats:
    pkl_path: Path
    total_frames: int
    valid_frames: int
    invalid_frames: int
    invalid_ratio: float


def as_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.size == 0 or arr.shape != (68, 2):
        return None
    return arr


def extract_frames(obj: Any) -> List[np.ndarray | None]:
    frames: List[np.ndarray | None] = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and "landmarks" in item:
                frames.append(as_array(item["landmarks"]))
            else:
                frames.append(as_array(item))
        return frames

    if isinstance(obj, dict):
        if "landmarks" in obj:
            for item in obj["landmarks"]:
                frames.append(as_array(item))
            return frames

        if "frames" in obj:
            for item in obj["frames"]:
                if isinstance(item, dict) and "landmarks" in item:
                    frames.append(as_array(item["landmarks"]))
                else:
                    frames.append(as_array(item))
            return frames

        try:
            keys = sorted(obj.keys(), key=lambda k: int(k))
        except Exception:
            keys = list(obj.keys())

        for key in keys:
            item = obj[key]
            if isinstance(item, dict) and "landmarks" in item:
                frames.append(as_array(item["landmarks"]))
            else:
                frames.append(as_array(item))
        return frames

    raise TypeError(f"Unsupported pkl root type: {type(obj)}")


def iter_pkl_files(root: Path) -> Iterable[Path]:
    return root.rglob("*.pkl")


def analyze_one_file(pkl_path: Path) -> FileStats:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    frames = extract_frames(obj)
    total = len(frames)
    valid = sum(1 for x in frames if x is not None)
    invalid = total - valid
    ratio = (invalid / total) if total > 0 else 1.0

    return FileStats(
        pkl_path=pkl_path,
        total_frames=total,
        valid_frames=valid,
        invalid_frames=invalid,
        invalid_ratio=ratio,
    )


def write_csv(rows: list[FileStats], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pkl_path",
                "total_frames",
                "valid_frames",
                "invalid_frames",
                "invalid_ratio",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    str(r.pkl_path),
                    r.total_frames,
                    r.valid_frames,
                    r.invalid_frames,
                    f"{r.invalid_ratio:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan landmark PKL files and report how many frames are invalid "
            "(None or wrong shape), including counts with and without a ratio cutoff."
        )
    )
    parser.add_argument(
        "--landmark-root",
        type=Path,
        default=Path(r"E:\lrs3_rj\lrs3\landmark\trainval"),
        help="Root folder that contains speaker subfolders with .pkl files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Invalid-frame ratio threshold used for fail count (default: 0.40).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save per-file stats CSV.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many worst files (highest invalid ratio) to print.",
    )
    args = parser.parse_args()

    root = args.landmark_root
    if not root.exists():
        raise FileNotFoundError(f"landmark root not found: {root}")

    file_stats: list[FileStats] = []
    unreadable: list[tuple[Path, str]] = []

    for pkl_path in iter_pkl_files(root):
        try:
            file_stats.append(analyze_one_file(pkl_path))
        except Exception as e:
            unreadable.append((pkl_path, f"{type(e).__name__}: {e}"))

    total_pkls = len(file_stats)
    total_frames = sum(s.total_frames for s in file_stats)
    total_invalid_frames = sum(s.invalid_frames for s in file_stats)
    total_valid_frames = sum(s.valid_frames for s in file_stats)

    any_missing = [s for s in file_stats if s.invalid_frames > 0]
    above_threshold = [s for s in file_stats if s.invalid_ratio > args.threshold]
    no_valid = [s for s in file_stats if s.valid_frames == 0]

    overall_invalid_ratio = (total_invalid_frames / total_frames) if total_frames > 0 else 0.0

    print("=" * 72)
    print("Landmark PKL Missing-Frame Analysis")
    print("=" * 72)
    print(f"Root:                           {root}")
    print(f"Threshold:                      {args.threshold:.2%}")
    print(f"Readable PKL files:             {total_pkls}")
    print(f"Unreadable PKL files:           {len(unreadable)}")
    print(f"PKL files with any missing:     {len(any_missing)}")
    print(f"PKL files above threshold:      {len(above_threshold)}")
    print(f"PKL files with zero valid:      {len(no_valid)}")
    print("-" * 72)
    print(f"Total frames (all readable):    {total_frames}")
    print(f"Total valid frames:             {total_valid_frames}")
    print(f"Total invalid frames:           {total_invalid_frames}")
    print(f"Overall invalid-frame ratio:    {overall_invalid_ratio:.2%}")

    if total_pkls > 0:
        print("-" * 72)
        print(f"Share of files with any missing: {(len(any_missing) / total_pkls):.2%}")
        print(f"Share of files above threshold:  {(len(above_threshold) / total_pkls):.2%}")

    if args.top > 0 and file_stats:
        print("-" * 72)
        print(f"Top {args.top} worst files by invalid ratio:")
        worst = sorted(
            file_stats,
            key=lambda s: (s.invalid_ratio, s.invalid_frames, s.total_frames),
            reverse=True,
        )[: args.top]
        for s in worst:
            print(
                f"  {s.invalid_frames}/{s.total_frames} ({s.invalid_ratio:.2%}) - {s.pkl_path}"
            )

    if unreadable:
        print("-" * 72)
        print("Unreadable PKL examples (up to 20):")
        for p, err in unreadable[:20]:
            print(f"  {p} -> {err}")

    if args.csv is not None:
        write_csv(file_stats, args.csv)
        print("-" * 72)
        print(f"Saved CSV: {args.csv}")

    print("=" * 72)


if __name__ == "__main__":
    main()
