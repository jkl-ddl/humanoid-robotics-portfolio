"""Steering locomotion configuration for T800."""

from __future__ import annotations

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.robots.t800 import (
  T800_ACTION_SCALE,
  T800_EE_BODY_NAMES,
  get_t800_robot_cfg,
)
from smp.rl.tasks.steering import mdp

T800_FOOT_BODY_NAMES = ("LINK_FOOT_L", "LINK_FOOT_R")
T800_FOOT_SITE_NAMES = ("force_sensor_left_foot", "force_sensor_right_foot")


def _add_t800_foot_sensors(cfg: ManagerBasedRlEnvCfg) -> None:
  feet_contact = ContactSensorCfg(
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
  foot_height = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=site_name, entity="robot")
      for site_name in T800_FOOT_SITE_NAMES
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_contact, foot_height)


def t800_steering_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = g1_smp_env_cfg(
    play=play,
    robot_cfg=get_t800_robot_cfg(),
    action_scale=T800_ACTION_SCALE,
    root_body_name="LINK_BASE",
    self_collision_body_pattern="LINK_BASE",
    ee_body_names=T800_EE_BODY_NAMES,
    num_joints=25,
  )
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 8000
  cfg.sim.contact_sensor_maxmatch = 256

  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=True,
    rand_face_dir=True,
    tar_speed_min=0.5,
    tar_speed_max=2.0,
    debug_vis=True,
  )
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          0.5,
          {"command_name": "steering", "vel_err_scale": 1.0},
        ),
        (mdp.steering_face_direction, 0.5, {"command_name": "steering"}),
      ),
    },
  )
  cfg.events["init_smp_state"].params["ckpt_path"] = os.environ.get(
    "SMP_T800_PRIOR",
    "datasets/pretrain_ckpt/pretrained_t800_loco.pt",
  )
  cfg.events.pop("foot_friction", None)
  cfg.events["base_com"].params["asset_cfg"].body_names = "LINK_WAIST_YAW"
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.72,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  return cfg


def t800_natural_steering_smp_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Steering A/B with aligned facing and official MjLab gait costs."""
  cfg = t800_steering_smp_env_cfg(play=play)
  command = cfg.commands["steering"]
  command.rand_face_dir = False
  command.tar_speed_min = 0.5
  command.tar_speed_max = 1.2
  _add_t800_foot_sensors(cfg)

  cfg.rewards["task_smp_product"].params["task_terms"] = (
    (
      mdp.steering_target_velocity,
      0.55,
      {"command_name": "steering", "vel_err_scale": 1.0},
    ),
    (mdp.steering_face_direction, 0.45, {"command_name": "steering"}),
  )
  cfg.rewards["flat_orientation"] = RewardTermCfg(
    func=envs_mdp.flat_orientation_l2,
    weight=-0.15,
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=velocity_mdp.feet_clearance,
    weight=-0.10,
    params={
      "target_height": 0.10,
      "height_sensor_name": "foot_height_scan",
      "asset_cfg": SceneEntityCfg("robot", site_names=T800_FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=velocity_mdp.feet_slip,
    weight=-0.05,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "steering",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=T800_FOOT_SITE_NAMES),
    },
  )
  cfg.rewards["action_rate"] = RewardTermCfg(
    func=envs_mdp.action_rate_l2,
    weight=-0.001,
  )
  cfg.rewards["joint_limit"] = RewardTermCfg(
    func=envs_mdp.joint_pos_limits,
    weight=-0.05,
  )
  return cfg
