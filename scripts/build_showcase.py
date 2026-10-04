"""Build this portfolio from archived evidence, or replot its public CSV files.

No training, policy execution, motion editing, or source-repository changes.
Requires numpy, matplotlib, tensorboard, opencv-python, imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAGS = {
    "mean_reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",
    "body_pos_error": "Metrics/motion/error_body_pos",
    "body_rot_error": "Metrics/motion/error_body_rot",
    "joint_pos_error": "Metrics/motion/error_joint_pos",
    "joint_vel_error": "Metrics/motion/error_joint_vel",
    "anchor_pos_error": "Metrics/motion/error_anchor_pos",
    "anchor_rot_error": "Metrics/motion/error_anchor_rot",
    "velocity_error": "Metrics/steering/error_vel_xy",
    "facing_error": "Metrics/steering/error_face",
    "slip_logged": "Metrics/slip_velocity_mean",
    "upright_reward": "Episode_Reward/flat_orientation",
    "joint_limit_reward": "Episode_Reward/joint_limit",
    "time_out": "Episode_Termination/time_out",
    "ee_body_pos_termination": "Episode_Termination/ee_body_pos",
    "policy_std": "Policy/mean_std",
    "value_loss": "Loss/value",
    "surrogate_loss": "Loss/surrogate",
    "fps": "Perf/total_fps",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scalar_columns(source: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(source), size_guidance={"scalars": 0}).Reload()
    available = set(acc.Tags()["scalars"])
    columns = {key: {int(v.step): float(v.value) for v in acc.Scalars(tag)}
               for key, tag in TAGS.items() if tag in available}
    if not columns or "mean_reward" not in columns:
        raise ValueError(f"Missing training scalars: {source.name}")
    return columns


def write_scalars(columns: dict, name: str) -> dict:
    steps = sorted(set().union(*(series.keys() for series in columns.values())))
    with (ROOT / "evidence" / f"{name}_training.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", *columns])
        for step in steps:
            writer.writerow([step, *[format(series[step], ".9g") if step in series else ""
                                    for series in columns.values()]])
    return {
        "iteration_first": steps[0], "iteration_last": steps[-1],
        "logged_iterations": len(steps), "scalar_columns": {k: TAGS[k] for k in columns},
        "last_100_logged_values_mean": {k: float(np.mean(list(v.values())[-100:]))
                                        for k, v in columns.items()},
    }


def collect_scalars(source: Path, name: str) -> dict:
    return write_scalars(scalar_columns(source), name)


def collect_resumed_curves(original: Path, resumed: list[Path], name: str) -> dict:
    columns = scalar_columns(original)
    seams = []
    # Retain only the selected checkpoint's history. Later abandoned updates
    # from an older run must not be joined into the resumed branch's curve.
    for run in resumed:
        incoming = scalar_columns(run)
        start = min(incoming["mean_reward"])
        columns = {key: {step: value for step, value in values.items() if step < start}
                   for key, values in columns.items()}
        for key, values in incoming.items():
            columns.setdefault(key, {}).update(values)
        seams.append({"iteration": start, "run": run.name})
    result = write_scalars(columns, name)
    result["resume_seams"] = seams
    result["history_note"] = "Selected resume branch only; repeated iteration labels use the later run, and abandoned updates are excluded."
    return result


def add_completed_taunt(args: argparse.Namespace) -> None:
    if not args.taunt_original_run or not args.taunt_resume_run or not args.taunt_eval:
        raise ValueError("--add-taunt requires original run, resume runs and validated evaluation directory")
    acceptance = json.loads((args.taunt_eval / "taunt_acceptance.json").read_text(encoding="utf-8"))
    if not (acceptance["requested_frames"] == 1190
            and acceptance["continuous_frames_before_first_termination"] == 1190
            and acceptance["first_termination"] is None
            and acceptance["failure_resets_disabled"] == []
            and acceptance["start_frame"] == 0
            and acceptance["all_recorded_states_finite"]):
        raise ValueError("Taunt did not pass the requested full 1190-frame protocol")
    index_path = ROOT / "evidence" / "artifact_index.json"
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    manifest.pop("training_rerun", None)
    manifest["collector_launches_training"] = False
    manifest.setdefault("closeout_training", {})["taunt"] = "additional native 1024-env continuation"
    manifest["curves"]["taunt"] = collect_resumed_curves(args.taunt_original_run, args.taunt_resume_run, "taunt")
    manifest["videos"]["taunt"] = copy_video(
        args.taunt_eval / "taunt_policy_rollout.mp4", "taunt", 0, 23.8)
    if manifest["videos"]["taunt"]["frames"] != 1190:
        raise ValueError("Video frame count does not match the accepted continuous trajectory")
    shutil.copyfile(args.taunt_eval / "taunt_acceptance.json", ROOT / "evidence" / "taunt_acceptance.json")
    shutil.copyfile(args.taunt_eval / "taunt_policy_states.npz", ROOT / "evidence" / "taunt_policy_states.npz")
    write_json(index_path, manifest)


def add_completed_steering(args: argparse.Namespace) -> None:
    if not (args.steering_run and args.steering_resume_run and args.steering_eval and args.steering_video):
        raise ValueError("--add-steering requires original run, resume runs, evaluation and reviewed video")
    metrics = json.loads((args.steering_eval / "metrics.json").read_text(encoding="utf-8"))
    if not (metrics["requested_steps"] == 1000
            and metrics["continuous_steps_before_first_termination"] == 1000
            and metrics["first_termination"] is None
            and metrics["command_observation_refreshed_before_policy"]
            and metrics["command_observation_max_error"] < 1e-6):
        raise ValueError("Steering did not pass the continuous command-observation protocol")
    index_path = ROOT / "evidence" / "artifact_index.json"
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    manifest.setdefault("closeout_training", {})["steering"] = "512-env quality continuation; see config boundaries"
    manifest["curves"]["steering"] = collect_resumed_curves(
        args.steering_run, args.steering_resume_run, "steering")
    manifest["videos"]["steering"] = copy_video(args.steering_video, "steering", 0, 20)
    if manifest["videos"]["steering"]["frames"] != 1000:
        raise ValueError("Video frame count does not match the continuous trajectory")
    metrics["checkpoint"] = Path(metrics["checkpoint"]).name
    metrics["protocol_note"] = {
        "single_trace_only": True,
        "direction_frame": "world; -x means travel direction, not backward gait",
        "observation_noise": "unchanged from the task play configuration",
        "rendering": "recorded policy states; no reference-motion replacement or reset splice",
    }
    write_json(ROOT / "evidence" / "steering_metrics.json", metrics)
    shutil.copyfile(args.steering_eval / "rollout_metrics.npz", ROOT / "evidence" / "steering_rollout.npz")
    shutil.copyfile(args.steering_eval / "policy_states.npz", ROOT / "evidence" / "steering_policy_states.npz")
    write_json(index_path, manifest)


def copy_video(source: Path, name: str, preview_start: float, preview_duration: float) -> dict:
    import cv2
    import imageio_ffmpeg

    destination = ROOT / "media" / f"{name}.mp4"
    shutil.copyfile(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError(f"Video copy mismatch: {name}")
    cap = cv2.VideoCapture(str(destination))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0 or fps <= 0:
        raise ValueError(f"Invalid video: {name}")
    frames = []
    for index in np.linspace(0, n - 1, 12, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode {name} frame {index}")
        frame = cv2.resize(frame, (384, round(height * 384 / width)))
        cv2.putText(frame, f"{index / fps:.2f}s", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 80, 255), 1, cv2.LINE_AA)
        frames.append(frame)
    cap.release()
    sheet = np.vstack([np.hstack(frames[i:i + 4]) for i in range(0, 12, 4)])
    if not cv2.imwrite(str(ROOT / "media" / f"{name}_frames.jpg"), sheet):
        raise RuntimeError("Could not save contact sheet")
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(preview_start), "-i", str(destination), "-t", str(preview_duration),
        "-filter_complex", "fps=8,scale=480:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        "-loop", "0", str(ROOT / "media" / f"{name}.gif"),
    ], check=True)
    return {"path": f"media/{name}.mp4", "sha256": digest, "frames": n, "fps": fps,
            "duration_seconds": n / fps, "width": width, "height": height,
            "gif_preview_start_seconds": preview_start, "gif_preview_duration_seconds": preview_duration}


def collect(args: argparse.Namespace) -> None:
    required = ("steering_run", "steering_eval", "celebration_run", "celebration_eval", "badminton_root")
    if any(getattr(args, key) is None for key in required):
        raise ValueError("--collect requires the five source directory arguments")
    for directory in ("evidence", "media"):
        (ROOT / directory).mkdir(exist_ok=True)
    manifest: dict = {"packaged_on": "2026-10-03", "collector_launches_training": False, "seed": 42,
                      "curves": {}, "videos": {}}
    for name, run in (("steering", args.steering_run), ("celebration", args.celebration_run)):
        manifest["curves"][name] = collect_scalars(run, name)
    badminton_events = sorted(args.badminton_root.rglob("events.out.tfevents.*"))
    if len(badminton_events) != 1:
        raise ValueError("Expected one final-stage badminton event file")
    manifest["curves"]["badminton"] = collect_scalars(badminton_events[0], "badminton")

    steering = json.loads((args.steering_eval / "metrics.json").read_text(encoding="utf-8"))
    steering["checkpoint"] = Path(steering["checkpoint"]).name
    steering["protocol_note"] = {
        "seed": 42, "environments": 1, "transition_steps": 50,
        "transition_seconds": 1.0, "single_trace_only": True,
        "direction_frame": "world; backward label means -x travel, not backward gait",
        "source": "archived smooth50 candidate; no reevaluation during publication",
    }
    write_json(ROOT / "evidence" / "steering_metrics.json", steering)
    with np.load(args.steering_eval / "rollout_metrics.npz", allow_pickle=False) as rollout:
        # This archive contains numeric telemetry only; no machine paths or model weights.
        np.savez_compressed(ROOT / "evidence" / "steering_rollout.npz",
                            **{key: rollout[key] for key in rollout.files})
    acceptance = args.badminton_root / "beyondmimic" / "acceptance"
    shutil.copyfile(acceptance / "rollout_model4493_perframe_start40_contact_cpu_acceptance.json",
                    ROOT / "evidence" / "badminton_acceptance.json")

    with np.load(args.celebration_eval / "celebration_model19999_policy_rollout_states.npz",
                 allow_pickle=False) as rollout:
        celebration = {
            "checkpoint": "model_19999.pt", "motion": "EngineAI dance_t800",
            "num_envs_training": 1024, "training_iterations": 20000, "seed": 42,
            "visualized_state_frames": len(rollout["joint_pos"]),
            "joint_count": int(rollout["joint_pos"].shape[1]),
            "root_height_range_m": [float(rollout["root_pos"][:, 2].min()),
                                    float(rollout["root_pos"][:, 2].max())],
            "failure_resets_disabled": rollout["failure_resets_disabled"].tolist(),
            "classification": "trained-policy state visualization; see separate retained-termination acceptance",
            "visualization_protocol": "fixed motion frame 0; perturbations/noise off; extended episode; original failure terms retained if disabled list is empty",
        }
    write_json(ROOT / "evidence" / "celebration_summary.json", celebration)
    manifest["videos"]["steering"] = copy_video(
        args.steering_eval / "t800-steering-model_30499-step-0.mp4", "steering", 0, 20)
    manifest["videos"]["celebration"] = copy_video(
        args.celebration_eval / "celebration_model19999_policy_rollout.mp4", "celebration", 9, 8)
    manifest["videos"]["badminton"] = copy_video(
        acceptance / "badminton_model4493_perframe_start40_corrected.mp4", "badminton", 0, 4.72)
    write_json(ROOT / "evidence" / "artifact_index.json", manifest)


def plot_csv(name: str, panels: list[tuple[str, str, str]], title: str) -> None:
    with (ROOT / "evidence" / f"{name}_training.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.asarray([int(row["iteration"]) for row in rows])
    index_path = ROOT / "evidence" / "artifact_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    seams = [int(seam["iteration"]) for seam in
             index.get("curves", {}).get(name, {}).get("resume_seams", [])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), layout="constrained")
    for axis, (key, label, unit) in zip(axes.flat, panels):
        y = np.asarray([float(row[key]) if row.get(key) else np.nan for row in rows])
        axis.plot(x, y, color="#8ba5ba", alpha=0.40, linewidth=0.7, label="Logged value")
        smoothed = np.full_like(y, np.nan)
        for i in range(99, len(y)):
            window = y[i - 99:i + 1]
            crosses_seam = any(x[i - 99] < seam <= x[i] for seam in seams)
            if np.isfinite(window).all() and x[i] - x[i - 99] == 99 and not crosses_seam:
                smoothed[i] = window.mean()
        axis.plot(x, smoothed, color="#167b85", linewidth=1.7, label="Trailing 100 iterations")
        for j, seam in enumerate(seams):
            axis.axvline(seam, color="#7a6b60", linestyle="--", linewidth=0.8,
                         label="Resume / config boundary" if j == 0 else None)
        axis.set(title=label, xlabel="Logged PPO iteration", ylabel=unit)
        axis.grid(alpha=0.20)
        axis.spines[["top", "right"]].set_visible(False)
        if len(x) > 10000:
            axis.ticklabel_format(axis="x", style="plain")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(ROOT / "media" / f"{name}_training.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--add-taunt", action="store_true")
    parser.add_argument("--add-steering", action="store_true")
    parser.add_argument("--taunt-original-run", type=Path)
    parser.add_argument("--taunt-resume-run", type=Path, action="append")
    parser.add_argument("--taunt-eval", type=Path)
    parser.add_argument("--steering-resume-run", type=Path, action="append")
    parser.add_argument("--steering-video", type=Path)
    for name in ("steering_run", "steering_eval", "celebration_run", "celebration_eval", "badminton_root"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    if args.collect:
        collect(args)
    if args.add_taunt:
        add_completed_taunt(args)
    if args.add_steering:
        add_completed_steering(args)
    plot_csv("celebration", [
        ("mean_reward", "Episode reward", "Logged reward"),
        ("episode_length", "Episode length", "Control steps"),
        ("body_pos_error", "Body position error (mean body norm)", "m"),
        ("joint_pos_error", "Joint position error (25-joint L2 norm)", "rad"),
    ], "T800 celebration | 1024 envs, seed 42 | iterations 0-19999")
    plot_csv("steering", [
        ("mean_reward", "Episode reward", "Logged reward"),
        ("episode_length", "Episode length", "Control steps"),
        ("velocity_error", "Command velocity error", "m/s (logger definition)"),
        ("facing_error", "Facing error", "Logged error"),
    ], "T800 natural steering | 512 envs, seed 42 | selected warm-start history")
    plot_csv("badminton", [
        ("mean_reward", "Episode reward", "Logged reward"),
        ("episode_length", "Episode length", "Control steps"),
        ("body_pos_error", "Body position error", "m (logger definition)"),
        ("joint_pos_error", "Joint position error", "rad (logger definition)"),
    ], "G1 badminton-style tracking | recorded final stage, iterations 3994-4493")
    if (ROOT / "evidence" / "taunt_training.csv").exists():
        plot_csv("taunt", [
            ("mean_reward", "Episode reward", "Logged reward"),
            ("episode_length", "Episode length", "Control steps"),
            ("body_pos_error", "Body position error (mean body norm)", "m"),
            ("joint_pos_error", "Joint position error (25-joint L2 norm)", "rad"),
        ], "T800 rapid punch | 1024 envs, seed 42 | selected resume branch")
    print("Built figures from archived scalars; no robot policy was run.")


if __name__ == "__main__":
    main()
