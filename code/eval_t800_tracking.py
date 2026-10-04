"""Finite fixed-start EngineAI tracking evaluation with failure terms retained.

Use inside the pinned EngineAI/Isaac Lab runtime. No cameras or training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Tracking-Flat-T800-Wo-State-Estimation-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--motion-file", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--num-steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args, hydra = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra]
app = AppLauncher(args).app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner
import engineai_rl_lab.tasks  # noqa: F401


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite evaluation evidence")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    env_cfg.commands.motion.motion_file = str(args.motion_file.resolve())
    for key in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        if hasattr(env_cfg.events, key):
            setattr(env_cfg.events, key, None)
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    for group in ("policy", "actor"):
        if hasattr(env_cfg.observations, group):
            getattr(env_cfg.observations, group).enable_corruption = False
    with np.load(args.motion_file, allow_pickle=False) as motion:
        length = int(motion["joint_pos"].shape[0])
    steps = args.num_steps or (length - args.start_frame)
    dt = env_cfg.decimation * env_cfg.sim.dt
    # Only the time horizon is extended to fit one complete motion; failure
    # thresholds and functions are deliberately unchanged.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, steps * dt + dt)
    raw = gym.make(args.task, cfg=env_cfg)
    command = raw.unwrapped.command_manager.get_term("motion")
    if not 0 <= args.start_frame < length:
        raise ValueError("start-frame outside reference motion")
    original_sampling = command._adaptive_sampling

    def fixed_start(env_ids):
        command.time_steps[env_ids] = args.start_frame

    command._adaptive_sampling = fixed_start
    # The RSL wrapper resets the environment in its constructor. Create it
    # BEFORE the final fixed reset/cache refresh, otherwise it invalidates the
    # freshly aligned reference cache again.
    env = RslRlVecEnvWrapper(raw)
    raw.reset()
    # Reset writes robot poses but leaves the relative-reference cache stale.
    # Refresh that cache using the actual command implementation at the SAME
    # reference frame, before any policy step or termination evaluation.
    command.time_steps -= 1
    command._update_command()
    command._update_metrics()
    raw.unwrapped.observation_manager.compute()
    command._adaptive_sampling = original_sampling
    cfg = agent_cfg.to_dict()
    allowed = {"class_name", "hidden_dims", "activation", "obs_normalization", "distribution_cfg"}
    for key in ("actor", "critic"):
        if isinstance(cfg.get(key), dict):
            cfg[key] = {k: v for k, v in cfg[key].items() if k in allowed}
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=raw.unwrapped.device)
    obs = env.get_observations()
    robot = raw.unwrapped.scene["robot"]
    terms = tuple(raw.unwrapped.termination_manager.active_terms)
    records = {key: [] for key in ("joint_pos", "root_pos", "root_quat", "done", "motion_frame")}
    errors = {key: [] for key in ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_joint_pos")}
    first_termination = None

    def record():
        records["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        records["root_pos"].append(robot.data.root_pos_w[0].cpu().numpy().copy())
        records["root_quat"].append(robot.data.root_quat_w[0].cpu().numpy().copy())
        records["motion_frame"].append(int(command.time_steps[0].item()))
        records["done"].append(False)
        command._update_metrics()
        for key in errors:
            errors[key].append(float(command.metrics[key][0].item()))

    with torch.inference_mode():
        record()
        for step in range(1, steps):
            obs, _, dones, _ = env.step(policy(obs))
            active = [key for key in terms if bool(raw.unwrapped.termination_manager.get_term(key)[0].item())]
            if bool(torch.as_tensor(dones).reshape(-1)[0]):
                first_termination = {"step": step, "terms": active}
                # env.step has already autoreset the robot; do not render that
                # reset pose as the last pose of the continuous trajectory.
                break
            record()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{k: np.asarray(v) for k, v in records.items()},
                        **{k: np.asarray(v) for k, v in errors.items()},
                        joint_names=np.asarray(robot.joint_names),
                        checkpoint=np.asarray(args.checkpoint.name),
                        motion_file=np.asarray(args.motion_file.name), seed=np.asarray(args.seed),
                        failure_resets_disabled=np.asarray([], dtype="U1"))
    result = {
        "checkpoint": args.checkpoint.name, "motion": args.motion_file.name,
        "seed": args.seed, "start_frame": args.start_frame, "requested_frames": steps,
        "continuous_frames_before_first_termination": len(records["root_pos"]),
        "control_dt": dt, "first_termination": first_termination,
        "active_termination_terms": terms, "failure_resets_disabled": [],
        "reset_cache_refreshed_at_same_reference_frame": True,
        "time_horizon_extended_to_one_motion": True,
        "error_max": {k: float(np.max(v)) for k, v in errors.items()},
        "error_mean": {k: float(np.mean(v)) for k, v in errors.items()},
        "all_recorded_states_finite": all(np.isfinite(np.asarray(records[k])).all()
                                         for k in ("root_pos", "root_quat", "joint_pos")),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("EVALUATION_RESULT " + json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
