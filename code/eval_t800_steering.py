"""Finite deterministic command-response evaluation for Smp-Steering-T800."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import smp.rl.tasks  # noqa: F401

FOOT_BODY_NAMES = ("LINK_FOOT_L", "LINK_FOOT_R")
FOOT_SITE_NAMES = ("force_sensor_left_foot", "force_sensor_right_foot")
TORSO_BODY_NAME = "LINK_WAIST_YAW"
COMMANDS = (
  ("forward_0.5", 0.0, 0.5),
  ("forward_1.0", 0.0, 1.0),
  ("left_0.75", math.pi / 2, 0.75),
  ("backward_0.75", math.pi, 0.75),
  ("right_0.75", -math.pi / 2, 0.75),
)


def select_commands(names: list[str] | None) -> tuple[tuple[str, float, float], ...]:
  """Return the requested deterministic command segments in caller order."""
  if names is None:
    return COMMANDS
  by_name = {command[0]: command for command in COMMANDS}
  return tuple(by_name[name] for name in names)


def interpolate_command(
  start_angle: float,
  start_speed: float,
  target_angle: float,
  target_speed: float,
  step: int,
  transition_steps: int,
) -> tuple[float, float]:
  """Rate-limit one command transition using the shortest heading arc."""
  if transition_steps <= 0:
    return target_angle, target_speed
  alpha = min((step + 1) / transition_steps, 1.0)
  angle_delta = math.atan2(
    math.sin(target_angle - start_angle), math.cos(target_angle - start_angle)
  )
  return (
    start_angle + alpha * angle_delta,
    start_speed + alpha * (target_speed - start_speed),
  )


def summarize_segment(
  name: str, speed: float, rows: list[dict[str, object]]
) -> dict[str, object]:
  """Summarize one fixed-command segment with gait and posture metrics."""
  foot_height = np.asarray([row["foot_height"] for row in rows], dtype=np.float64)
  clearance = np.maximum(foot_height - foot_height.min(axis=0, keepdims=True), 0.0)
  foot_pos_xy = np.asarray([row["foot_pos_xy"] for row in rows], dtype=np.float64)
  target_dir = np.asarray([row["target_dir"] for row in rows], dtype=np.float64)
  foot_projection = np.sum(foot_pos_xy * target_dir[:, None, :], axis=-1)
  foot_separation = np.abs(foot_projection[:, 0] - foot_projection[:, 1])
  foot_speed = np.asarray([row["foot_speed_xy"] for row in rows], dtype=np.float64)
  foot_contact = np.asarray([row["foot_contact"] for row in rows], dtype=np.bool_)
  contact_slip = [
    float(foot_speed[:, index][foot_contact[:, index]].mean())
    if foot_contact[:, index].any()
    else None
    for index in range(2)
  ]
  return {
    "segment": name,
    "command_speed": speed,
    "mean_projected_speed": float(
      np.mean([float(row["projected_speed"]) for row in rows])
    ),
    "mean_velocity_error": float(
      np.mean([float(row["velocity_error"]) for row in rows])
    ),
    "mean_face_dot": float(np.mean([float(row["face_dot"]) for row in rows])),
    "termination_count": int(sum(bool(row["done"]) for row in rows)),
    "torso_pitch_mean_deg": float(
      np.mean([float(row["torso_pitch_deg"]) for row in rows])
    ),
    "torso_pitch_abs_p95_deg": float(
      np.quantile(np.abs([float(row["torso_pitch_deg"]) for row in rows]), 0.95)
    ),
    "torso_roll_abs_p95_deg": float(
      np.quantile(np.abs([float(row["torso_roll_deg"]) for row in rows]), 0.95)
    ),
    "foot_clearance_p95_m": np.quantile(clearance, 0.95, axis=0).tolist(),
    "foot_separation_command_p95_m": float(np.quantile(foot_separation, 0.95)),
    "foot_contact_fraction": foot_contact.mean(axis=0).tolist(),
    "both_feet_contact_fraction": float(foot_contact.all(axis=1).mean()),
    "foot_contact_slip_mean_mps": contact_slip,
  }


def _add_eval_foot_sensors(env_cfg) -> None:
  sensor_names = {sensor.name for sensor in (env_cfg.scene.sensors or ())}
  additions = []
  if "feet_ground_contact" not in sensor_names:
    additions.append(
      ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
          mode="subtree",
          pattern=r"^(LINK_FOOT_L|LINK_FOOT_R)$",
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      )
    )
  if "foot_height_scan" not in sensor_names:
    additions.append(
      TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
          ObjRef(type="site", name=site_name, entity="robot")
          for site_name in FOOT_SITE_NAMES
        ),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
      )
    )
  env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + tuple(additions)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--steps-per-command", type=int, default=200)
  parser.add_argument(
    "--transition-steps",
    type=int,
    default=0,
    help="Steps used to smoothly interpolate each command change.",
  )
  parser.add_argument("--task-id", default="Smp-Steering-T800")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--no-video", action="store_true", help="Save real policy states for separate rendering.")
  parser.add_argument(
    "--segments",
    nargs="+",
    choices=[command[0] for command in COMMANDS],
    help="Evaluate only the listed command segments, in the given order.",
  )
  args = parser.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  task_id = args.task_id
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = args.seed
  env_cfg.viewer.width = 960
  env_cfg.viewer.height = 720
  _add_eval_foot_sensors(env_cfg)

  commands = select_commands(args.segments)
  total_steps = len(commands) * args.steps_per_command
  if any((args.output_dir / name).exists() for name in ("metrics.json", "rollout_metrics.npz", "policy_states.npz")):
    raise FileExistsError("Refusing to overwrite evaluation evidence")
  # This test is externally commanded; do not insert random commands mid-test.
  duration = total_steps * env_cfg.decimation * env_cfg.sim.mujoco.timestep + 1.0
  env_cfg.commands["steering"].resampling_time_range = (duration, duration)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None if args.no_video else "rgb_array")
  video_env = raw_env if args.no_video else VideoRecorder(
    raw_env,
    video_folder=args.output_dir,
    step_trigger=lambda step: step == 0,
    video_length=total_steps,
    name_prefix=f"t800-steering-{args.checkpoint.stem}",
  )
  env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location="cuda:0",
  )
  policy = runner.get_inference_policy(device="cuda:0")
  steering = env.unwrapped.command_manager.get_term("steering")
  robot = env.unwrapped.scene["robot"]
  foot_site_ids, foot_site_names = robot.find_sites(
    FOOT_SITE_NAMES, preserve_order=True
  )
  if tuple(foot_site_names) != FOOT_SITE_NAMES:
    raise RuntimeError(
      f"T800 foot site mapping mismatch: {tuple(foot_site_names)} != {FOOT_SITE_NAMES}"
    )
  torso_body_ids, torso_body_names = robot.find_bodies(
    TORSO_BODY_NAME, preserve_order=True
  )
  if tuple(torso_body_names) != (TORSO_BODY_NAME,):
    raise RuntimeError(
      f"T800 torso mapping mismatch: {tuple(torso_body_names)} != {(TORSO_BODY_NAME,)}"
    )
  contact_sensor = env.unwrapped.scene["feet_ground_contact"]
  height_sensor = env.unwrapped.scene["foot_height_scan"]
  observation_manager = env.unwrapped.observation_manager
  actor_terms = observation_manager.active_terms["actor"]
  actor_dims = observation_manager.group_obs_term_dim["actor"]
  command_index = actor_terms.index("command")
  command_start = sum(math.prod(dim) for dim in actor_dims[:command_index])
  command_size = math.prod(actor_dims[command_index])
  if command_size != 5:
    raise ValueError(f"Expected the five-dimensional steering command, got {command_size}")
  command_slice = slice(command_start, command_start + command_size)
  command_observation_max_error = 0.0

  rows: list[dict[str, float | int | str | bool]] = []
  states = {name: [] for name in ("joint_pos", "root_pos", "root_quat_wxyz")}
  first_termination = None
  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(total_steps):
      segment = min(step // args.steps_per_command, len(commands) - 1)
      name, target_angle, target_speed = commands[segment]
      if segment == 0:
        start_angle, start_speed = target_angle, target_speed
      else:
        _, start_angle, start_speed = commands[segment - 1]
      angle, speed = interpolate_command(
        start_angle,
        start_speed,
        target_angle,
        target_speed,
        step % args.steps_per_command,
        args.transition_steps,
      )
      direction = torch.tensor(
        [math.cos(angle), math.sin(angle)], device=env.unwrapped.device
      )
      steering.tar_dir_w[0] = direction
      steering.face_dir_w[0] = direction
      steering.tar_speed[0] = speed
      steering._update_command()

      # compute() / get_observations() may return a cached concatenated group.
      # The actor has no history here: refresh this group without advancing the
      # critic's ten-frame history or taking a second physics step.
      obs["actor"] = observation_manager.compute_group("actor", update_history=False)
      actor_command = obs["actor"][:, command_slice]
      torch.testing.assert_close(actor_command, steering.command, rtol=0, atol=1e-6)
      command_observation_max_error = max(command_observation_max_error,
        float((actor_command - steering.command).abs().max().item()))
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      if bool(dones[0].item()):
        first_termination = {"step": step, "terms": [
          term for term in env.unwrapped.termination_manager.active_terms
          if bool(env.unwrapped.termination_manager.get_term(term)[0].item())
        ]}
        # The failing step has already autoreset. Never export its reset state
        # as part of the continuous trajectory.
        break
      states["joint_pos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
      states["root_pos"].append(robot.data.root_link_pos_w[0].detach().cpu().numpy().copy())
      states["root_quat_wxyz"].append(robot.data.root_link_quat_w[0].detach().cpu().numpy().copy())
      velocity = robot.data.root_link_lin_vel_w[0, :2]
      heading = robot.data.heading_w[0]
      face = torch.stack((torch.cos(heading), torch.sin(heading)))
      torso_quat = robot.data.body_link_quat_w[0, torso_body_ids, :]
      torso_roll, torso_pitch, _ = euler_xyz_from_quat(torso_quat)
      foot_velocity = robot.data.site_lin_vel_w[0, foot_site_ids, :2]
      foot_position = robot.data.site_pos_w[0, foot_site_ids, :2]
      if contact_sensor.data.found is None:
        raise RuntimeError("feet_ground_contact did not produce contact state")
      rows.append(
        {
          "step": step,
          "segment": name,
          "command_speed": speed,
          "velocity_x": float(velocity[0].item()),
          "velocity_y": float(velocity[1].item()),
          "projected_speed": float(torch.dot(velocity, direction).item()),
          "velocity_error": float(
            torch.linalg.vector_norm(velocity - speed * direction).item()
          ),
          "face_dot": float(torch.dot(face, direction).item()),
          "done": bool(dones[0].item()),
          "torso_roll_deg": math.degrees(float(torso_roll[0].item())),
          "torso_pitch_deg": math.degrees(float(torso_pitch[0].item())),
          "foot_height": height_sensor.data.heights[0].detach().cpu().tolist(),
          "foot_pos_xy": foot_position.detach().cpu().tolist(),
          "foot_speed_xy": torch.linalg.vector_norm(foot_velocity, dim=-1)
          .detach()
          .cpu()
          .tolist(),
          "foot_contact": (contact_sensor.data.found[0] > 0)
          .detach()
          .cpu()
          .tolist(),
          "target_dir": direction.detach().cpu().tolist(),
        }
      )

  env.close()
  metrics: list[dict[str, object]] = []
  for name, _, speed in commands:
    segment_rows = [row for row in rows if row["segment"] == name]
    if segment_rows:
      metrics.append(summarize_segment(name, speed, segment_rows))
  output = {
    "checkpoint": str(args.checkpoint.resolve()),
    "seed": args.seed,
    "step_dt": float(env.unwrapped.step_dt),
    "steps_per_command": args.steps_per_command,
    "transition_steps": args.transition_steps,
    "first_termination": first_termination,
    "continuous_steps_before_first_termination": len(rows),
    "requested_steps": total_steps,
    "command_observation_refreshed_before_policy": True,
    "command_observation_max_error": command_observation_max_error,
    "random_command_resampling_disabled": True,
    "segments": metrics,
  }
  (args.output_dir / "metrics.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  np.savez_compressed(
    args.output_dir / "rollout_metrics.npz",
    step=np.asarray([row["step"] for row in rows]),
    segment=np.asarray([row["segment"] for row in rows]),
    command_speed=np.asarray([row["command_speed"] for row in rows]),
    velocity=np.asarray([[row["velocity_x"], row["velocity_y"]] for row in rows]),
    projected_speed=np.asarray([row["projected_speed"] for row in rows]),
    velocity_error=np.asarray([row["velocity_error"] for row in rows]),
    face_dot=np.asarray([row["face_dot"] for row in rows]),
    torso_roll_deg=np.asarray([row["torso_roll_deg"] for row in rows]),
    torso_pitch_deg=np.asarray([row["torso_pitch_deg"] for row in rows]),
    foot_height=np.asarray([row["foot_height"] for row in rows]),
    foot_pos_xy=np.asarray([row["foot_pos_xy"] for row in rows]),
    foot_speed_xy=np.asarray([row["foot_speed_xy"] for row in rows]),
    foot_contact=np.asarray([row["foot_contact"] for row in rows], dtype=np.bool_),
    target_dir=np.asarray([row["target_dir"] for row in rows]),
    done=np.asarray([row["done"] for row in rows], dtype=np.bool_),
  )
  np.savez_compressed(
    args.output_dir / "policy_states.npz",
    **{name: np.asarray(values) for name, values in states.items()},
    joint_names=np.asarray(robot.joint_names),
    root_quaternion_order=np.asarray("wxyz"),
    checkpoint=np.asarray(args.checkpoint.name),
    seed=np.asarray(args.seed),
  )
  print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
