"""Small T800 quality-tuning branch using unmodified MjLab reward formulas.

The model, prior, observations, commands, PD, collision and termination settings
remain those of Smp-Steering-Natural-T800. This is a practical combined tuning
branch, not an experiment proving the separate causal effect of each term.
"""

from mjlab.envs import mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, register_mjlab_task

import smp.rl.tasks  # noqa: F401

BASE_TASK = "Smp-Steering-Natural-T800"
TASK_ID = "Smp-Steering-Quality-T800"


def quality_env_cfg(play=False):
    cfg = load_env_cfg(BASE_TASK, play=play)
    cfg.rewards["flat_orientation"].weight = -0.30
    # The T800 force-sensor site is ~0.064 m above the plane during contact.
    # A raw site target of 0.15 m corresponds to about 0.086 m sole clearance,
    # not a 0.15 m lift. This adapts the existing reward's geometric convention.
    cfg.rewards["foot_clearance"].params["target_height"] = 0.15
    cfg.rewards["upper_arm_posture"] = RewardTermCfg(
        func=mdp.posture,
        weight=0.015,
        params={
            "std": {r".*SHOULDER_PITCH.*": 0.60,
                    r".*SHOULDER_ROLL.*": 0.35,
                    r".*SHOULDER_YAW.*": 0.35},
            "asset_cfg": SceneEntityCfg("robot", joint_names=r".*SHOULDER.*"),
        },
    )
    return cfg


agent_cfg = load_rl_cfg(BASE_TASK)
agent_cfg.experiment_name = "smp_steering_quality_t800"
agent_cfg.run_name = "quality_tuning"
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=quality_env_cfg(),
    play_env_cfg=quality_env_cfg(play=True),
    rl_cfg=agent_cfg,
)

if __name__ == "__main__":
    from mjlab.scripts.train import main
    main()
