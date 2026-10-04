#!/usr/bin/env python3
"""Apply explicit acceptance gates to a rollout archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if len(indices) else None


parser = argparse.ArgumentParser()
parser.add_argument("npz", type=Path)
parser.add_argument("--min-frames", type=int, default=200)
parser.add_argument("--max-anchor-z-m", type=float, default=0.30)
parser.add_argument("--max-anchor-ori-deg", type=float, default=30.0)
parser.add_argument("--out", type=Path)
args = parser.parse_args()

with np.load(args.npz, allow_pickle=False) as data:
    terms = {
        key.removeprefix("termination_"): first_true(data[key])
        for key in data.files
        if key.startswith("termination_")
    }
    frames = int(len(data["done"]))
    report = {
        "frames": frames,
        "first_done_frame": first_true(data["done"]),
        "first_termination_frame": terms,
        "max_anchor_error_z_m": float(np.max(data["anchor_error_z"])),
        "max_anchor_error_ori_deg": float(np.degrees(np.max(data["anchor_error_ori"]))),
    }

gates = {
    "minimum_continuous_frames": frames >= args.min_frames,
    "anchor_z_limit": report["max_anchor_error_z_m"] < args.max_anchor_z_m,
    "anchor_orientation_limit": report["max_anchor_error_ori_deg"] < args.max_anchor_ori_deg,
    "no_termination": all(frame is None for frame in terms.values()),
}
report["gates"] = gates
report["pass"] = all(gates.values())
text = json.dumps(report, indent=2)
print(text)
if args.out:
    args.out.write_text(text + "\n", encoding="utf-8")
raise SystemExit(0 if report["pass"] else 1)
