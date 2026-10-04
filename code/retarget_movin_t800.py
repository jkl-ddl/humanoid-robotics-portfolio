"""MOVIN BVH -> official EngineAI GMR -> T800 reference (NOT a policy rollout).

Uses EngineAI GMR 71e02c0's unchanged IK and T800 match table. The format bridge
only restores metre units after its centimetre-based BVH loader and aliases the
MOVIN upper-spine name. Global FK grounding follows exptech/g1-moves's existing
retarget_all.py approach, using sole box corners rather than geometry centres.
No zero-slip/contact constraint, new IK solver, or learned control is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

GMR_COMMIT = "71e02c0598f6d6c6fe1514bf1d69bc24e82f801e"
SOURCE_URL = "https://huggingface.co/datasets/exptech/g1-moves"


def adapt_movin_frames(frames: list[dict]) -> list[dict]:
    # FootMod and Foot can share the same input array. Do not scale in place.
    output = [{name: [value[0] * 100., value[1].copy()]
               for name, value in frame.items()} for frame in frames]
    for frame in output:
        if "Spine2" not in frame:
            frame["Spine2"] = [value.copy() for value in frame["Spine1"]]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    paths = {suffix: args.output_prefix.with_suffix(suffix)
             for suffix in (".npz", ".csv", ".json")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Refusing to overwrite reference candidates")

    import mujoco
    import general_motion_retargeting as gmr
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    repo = Path(gmr.__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
         "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision != GMR_COMMIT:
        raise ValueError(f"Expected pinned EngineAI GMR, got {revision}")
    frame_time = next(float(line.split(":", 1)[1])
                      for line in args.bvh.read_text().splitlines()
                      if line.startswith("Frame Time:"))
    if not np.isclose(frame_time, 1/60, rtol=1e-5):
        raise ValueError("This bridge expects the original 60 Hz MOVIN capture")
    frames, human_height = load_bvh_file(str(args.bvh), format="nokov")
    frames = adapt_movin_frames(frames)
    if not .5 < float(frames[0]["Hips"][0][2]) < 1.5:
        raise ValueError("Metre/centimetre bridge failed its standing-height check")
    gmr.ROBOT_XML_DICT["engineai_t800"] = args.xml.resolve()
    retargeter = gmr.GeneralMotionRetargeting(
        src_human="bvh_lafan1", tgt_robot="engineai_t800",
        actual_human_height=human_height, verbose=False,
    )
    # Converge the first input before recording; never discard capture frames.
    for _ in range(20):
        retargeter.retarget(frames[0])
    rows = []
    for index, frame in enumerate(frames):
        rows.append(retargeter.retarget(frame))
        if index % 200 == 0:
            print(f"RETARGET {index}/{len(frames)}", flush=True)
    qpos = np.asarray(rows)
    if qpos.shape != (len(frames), 32) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite 25-DoF T800 reference")
    model = retargeter.model
    names = [model.joint(index).name for index in range(1, model.njnt)]
    if not all(name.startswith(f"J{index:02d}_")
               for index, name in enumerate(names)):
        raise ValueError("Unexpected T800 CSV joint order")
    feet = [index for index in range(model.ngeom)
            if model.body(int(model.geom_bodyid[index])).name
            in ("LINK_FOOT_L", "LINK_FOOT_R")
            and model.geom_contype[index] != 0
            and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_BOX]
    if not feet:
        raise ValueError("Expected official T800 sole collision boxes")
    data = mujoco.MjData(model)
    bottoms = []
    for pose in qpos:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        bottoms.append([data.geom_xpos[g, 2]
                        - abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
                        for g in feet])
    bottoms = np.asarray(bottoms)
    offset = float(bottoms.min())
    qpos[:, 2] -= offset  # One constant Z translation; joints/XY/rotation unchanged.
    exceed = np.maximum(model.jnt_range[1:, 0] - qpos[:, 7:],
                        qpos[:, 7:] - model.jnt_range[1:, 1]).clip(min=0)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths[".npz"], root_pos=qpos[:, :3], root_quat_wxyz=qpos[:, 3:7],
        joint_pos=qpos[:, 7:], joint_names=np.asarray(names), fps=np.asarray(60.),
        artifact_kind=np.asarray("kinematic_reference_not_policy"),
    )
    # EngineAI's existing CSV converter expects xyzw; recorded NPZ uses wxyz.
    csv = np.column_stack((qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]))
    np.savetxt(paths[".csv"], csv, delimiter=",", fmt="%.10f")
    result = {
        "artifact_kind": "kinematic_reference_not_policy", "gmr_commit": revision,
        "source": SOURCE_URL, "source_license": "CC-BY-4.0",
        "bvh_sha256": hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
        "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        "frames": len(qpos), "fps": 60,
        "format_bridge": "metres after cm loader; Spine1 alias Spine2; ToeBase",
        "ground_shift_m": -offset, "minimum_sole_before_m": offset,
        "minimum_sole_after_m": float((bottoms-offset).min()),
        "lowest_sole_after_p50_p95_m": np.percentile(
            (bottoms-offset).min(axis=1), [50, 95]).tolist(),
        "max_joint_limit_excess_rad": float(exceed.max()),
        "contact_constraint": False, "policy_executed": False,
    }
    paths[".json"].write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
