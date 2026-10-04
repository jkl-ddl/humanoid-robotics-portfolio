#!/usr/bin/env python3
"""Basic, model-independent checks for a BeyondMimic motion NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("npz", type=Path)
parser.add_argument("--max-joint-step-rad", type=float, default=0.5)
parser.add_argument("--max-root-step-m", type=float, default=0.2)
args = parser.parse_args()

with np.load(args.npz, allow_pickle=False) as data:
    keys = list(data.files)
    report: dict[str, object] = {"keys": keys}
    for key in ("joint_pos", "body_pos_w", "body_quat_w"):
        if key in data:
            report[key + "_shape"] = list(data[key].shape)
    if "joint_pos" in data and len(data["joint_pos"]) > 1:
        report["max_joint_step_rad"] = float(np.abs(np.diff(data["joint_pos"], axis=0)).max())
    if "body_pos_w" in data and len(data["body_pos_w"]) > 1:
        report["max_body_step_m"] = float(np.linalg.norm(np.diff(data["body_pos_w"], axis=0), axis=-1).max())
    report["finite"] = all(np.isfinite(data[key]).all() for key in keys)

report["pass"] = bool(
    report["finite"]
    and report.get("max_joint_step_rad", 0.0) <= args.max_joint_step_rad
    and report.get("max_body_step_m", 0.0) <= args.max_root_step_m
)
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["pass"] else 1)
