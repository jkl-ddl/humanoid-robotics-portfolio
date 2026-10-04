"""EngineAI T800 MuJoCo asset and actuator configuration.

The MuJoCo XML is intentionally not vendored: it is shared with the Native
SDK. Set SMP_T800_XML, or place engineai_robotics_native_sdk beside this
repository, before constructing the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

T800_JOINT_NAMES: tuple[str, ...] = (
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_TORSO_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
  "J23_HEAD_PITCH",
  "J24_HEAD_YAW",
)

T800_EE_BODY_NAMES: tuple[str, ...] = (
  "LINK_ANKLE_ROLL_L",
  "LINK_ANKLE_ROLL_R",
  "LINK_WAIST_YAW",
  "LINK_WRIST_END_L",
  "LINK_WRIST_END_R",
)

_ARMATURE_Q300HL = 0.2427264
_ARMATURE_Q300H = 0.14110848
_ARMATURE_Q200H = 0.0448737
_ARMATURE_Q50H = 0.0354625
_ARMATURE_Q25H = 0.00671625


def _xml_candidates() -> tuple[Path, ...]:
  repo_root = Path(__file__).resolve().parents[4]
  native_root = Path(
    os.environ.get("ENGINEAI_NATIVE_SDK_ROOT", repo_root.parent / "engineai_robotics_native_sdk")
  )
  return (
    Path(os.environ["SMP_T800_XML"]) if os.environ.get("SMP_T800_XML") else Path(),
    native_root / "assets/resource/robot/t800/xml/serial_t800.xml",
    repo_root / "assets/t800/xml/serial_t800.xml",
  )


def get_t800_xml_path() -> Path:
  for candidate in _xml_candidates():
    if candidate.is_file():
      return candidate.resolve()
  searched = "\n".join(str(p) for p in _xml_candidates() if str(p))
  raise FileNotFoundError(
    "T800 MuJoCo XML was not found. Set SMP_T800_XML or clone "
    "engineai_robotics_native_sdk beside smp. Searched:\n" + searched
  )


def get_spec() -> mujoco.MjSpec:
  xml_path = get_t800_xml_path()
  spec = mujoco.MjSpec.from_file(str(xml_path))
  # MjSpec.attach() can change the base directory used for relative assets.
  # Resolve every referenced resource before MjLab attaches this entity to the
  # scene; otherwise valid Native SDK textures may be reported as empty files.
  for asset in (*spec.meshes, *spec.textures):
    if asset.file and not os.path.isabs(asset.file):
      asset.file = str((xml_path.parent / asset.file).resolve())
  # Native SDK XML contains torque motors for standalone SDK simulation.
  # MjLab adds its own position actuators from T800_ARTICULATION.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


def _actuator(
  names: tuple[str, ...],
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=names,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
  )


T800_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _actuator(
      ("J00_HIP_PITCH_L", "J06_HIP_PITCH_R", "J03_KNEE_PITCH_L", "J09_KNEE_PITCH_R"),
      stiffness=180.0,
      damping=5.0,
      effort_limit=415.0 * 0.95,
      armature=_ARMATURE_Q300HL,
    ),
    _actuator(
      ("J01_HIP_ROLL_L", "J07_HIP_ROLL_R"),
      stiffness=100.0,
      damping=3.0,
      effort_limit=370.0,
      armature=_ARMATURE_Q300H,
    ),
    _actuator(
      ("J02_HIP_YAW_L", "J08_HIP_YAW_R", "J12_TORSO_YAW"),
      stiffness=100.0,
      damping=3.0,
      effort_limit=222.0,
      armature=_ARMATURE_Q200H,
    ),
    _actuator(
      (
        "J04_ANKLE_PITCH_L",
        "J05_ANKLE_ROLL_L",
        "J10_ANKLE_PITCH_R",
        "J11_ANKLE_ROLL_R",
      ),
      stiffness=40.0,
      damping=2.0,
      effort_limit=160.0,
      armature=_ARMATURE_Q50H,
    ),
    _actuator(
      (
        "J13_SHOULDER_PITCH_L",
        "J14_SHOULDER_ROLL_L",
        "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L",
        "J18_SHOULDER_PITCH_R",
        "J19_SHOULDER_ROLL_R",
        "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R",
      ),
      stiffness=50.0,
      damping=0.3,
      effort_limit=160.0,
      armature=_ARMATURE_Q50H,
    ),
    _actuator(
      ("J17_ELBOW_YAW_L", "J22_ELBOW_YAW_R", "J23_HEAD_PITCH", "J24_HEAD_YAW"),
      stiffness=50.0,
      damping=0.3,
      effort_limit=52.0,
      armature=_ARMATURE_Q25H,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

T800_ACTION_SCALE: dict[str, float] = {
  "J00_HIP_PITCH_L": 0.5,
  "J01_HIP_ROLL_L": 0.2,
  "J02_HIP_YAW_L": 0.2,
  "J03_KNEE_PITCH_L": 0.5,
  "J04_ANKLE_PITCH_L": 0.5,
  "J05_ANKLE_ROLL_L": 0.2,
  "J06_HIP_PITCH_R": 0.5,
  "J07_HIP_ROLL_R": 0.2,
  "J08_HIP_YAW_R": 0.2,
  "J09_KNEE_PITCH_R": 0.5,
  "J10_ANKLE_PITCH_R": 0.5,
  "J11_ANKLE_ROLL_R": 0.2,
  "J12_TORSO_YAW": 0.2,
  "J13_SHOULDER_PITCH_L": 0.2,
  "J14_SHOULDER_ROLL_L": 0.2,
  "J15_SHOULDER_YAW_L": 0.05,
  "J16_ELBOW_PITCH_L": 0.2,
  "J17_ELBOW_YAW_L": 0.05,
  "J18_SHOULDER_PITCH_R": 0.2,
  "J19_SHOULDER_ROLL_R": 0.2,
  "J20_SHOULDER_YAW_R": 0.05,
  "J21_ELBOW_PITCH_R": 0.2,
  "J22_ELBOW_YAW_R": 0.05,
  "J23_HEAD_PITCH": 0.2,
  "J24_HEAD_YAW": 0.2,
}

T800_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.06),
  joint_pos={
    "J00_HIP_PITCH_L": -0.06,
    "J01_HIP_ROLL_L": 0.0,
    "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.12,
    "J04_ANKLE_PITCH_L": -0.06,
    "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.06,
    "J07_HIP_ROLL_R": 0.0,
    "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.12,
    "J10_ANKLE_PITCH_R": -0.06,
    "J11_ANKLE_ROLL_R": 0.0,
    "J12_TORSO_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0,
    "J19_SHOULDER_ROLL_R": -0.15,
    "J20_SHOULDER_YAW_R": 0.0,
    "J21_ELBOW_PITCH_R": -0.25,
    "J22_ELBOW_YAW_R": 0.0,
    "J23_HEAD_PITCH": 0.0,
    "J24_HEAD_YAW": 0.0,
  },
  joint_vel={".*": 0.0},
)


def get_t800_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=T800_HOME_KEYFRAME,
    collisions=(),
    spec_fn=get_spec,
    articulation=T800_ARTICULATION,
  )
