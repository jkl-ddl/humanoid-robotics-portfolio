"""G1 → T800 动作数据重定向。

把 G1 的 36 列 CSV（3 root_pos + 4 root_quat + 29 joints）映射到 T800 的
32 列（3 root_pos + 4 root_quat + 25 joints）。

关节映射规则（详见任务 a 拆解）：
- 腿 12 个：一一对应，直接抄
- 腰：G1 waist_yaw → T800 torso_yaw；waist_roll/pitch 忽略（T800 没有）
- 肩 6 个：一一对应
- 肘：G1 elbow 的正向弯曲反号后映射到 T800 elbow_pitch
- 前臂：G1 wrist_roll 与 T800 elbow_yaw 的关节轴、零位不同，保持中立位
- 腕：wrist_pitch/yaw 忽略（T800 没有）
- 头：G1 无头关节，T800 head_pitch/yaw 补 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# G1 29 关节的顺序见 smp 的 scripts/csv_to_npz.py 的 JOINT_NAMES。
# 该列表的第 i 个元素 = T800 第 i 个关节（J00~J24）对应 G1 的哪一列；
# None 表示 T800 该关节没有 G1 对应，补 0。
G1_COL_TO_T800 = [
  # 左腿（G1 列 0~5）
  0, 1, 2, 3, 4, 5,
  # 右腿（G1 列 6~11）
  6, 7, 8, 9, 10, 11,
  # 腰 yaw（G1 列 12）
  12,
  # 左臂肩；肘 pitch/yaw 在下方单独处理。
  15, 16, 17, None, None,
  # 右臂肩；肘 pitch/yaw 在下方单独处理。
  22, 23, 24, None, None,
  # 头 pitch / yaw（G1 无，补 0）
  None, None,
]

# G1 的 pelvis 站立高度约 0.78m，T800 的 LINK_BASE 约 1.06m。
# 把 root_pos 的 z 上抬这个差值，避免 T800 加载后脚陷进地面。
ROOT_Z_OFFSET = 1.06 - 0.783675
T800_ELBOW_PITCH_MIN = -2.286


def retarget_array(data: np.ndarray) -> np.ndarray:
  """Convert a G1 motion array into sign-correct, limit-safe T800 qpos."""
  if data.ndim == 1:
    data = data[None, :]
  if data.ndim != 2 or data.shape[1] != 36:
    raise ValueError(f"期望二维 36 列数组，实际 shape={data.shape}")

  root = data[:, :7].copy()  # root_pos(3) + root_quat(4, xyzw)
  root[:, 2] += ROOT_Z_OFFSET
  g1_joints = data[:, 7:]

  t800_joints = np.zeros((data.shape[0], 25), dtype=data.dtype)
  for t800_idx, g1_col in enumerate(G1_COL_TO_T800):
    if g1_col is not None:
      t800_joints[:, t800_idx] = g1_joints[:, g1_col]

  # G1 elbows bend in the positive direction; T800 elbows bend negative.
  # Negative G1 values are hyper-extension for this transfer and map to straight.
  max_bend = -T800_ELBOW_PITCH_MIN
  t800_joints[:, 16] = -np.clip(g1_joints[:, 18], 0.0, max_bend)
  t800_joints[:, 21] = -np.clip(g1_joints[:, 25], 0.0, max_bend)

  # Do not copy G1 wrist_roll into T800 elbow_yaw. Their axes and neutral
  # frames differ, which visibly flips the T800 forearms. Zero is neutral.
  t800_joints[:, 17] = 0.0
  t800_joints[:, 22] = 0.0

  return np.hstack([root, t800_joints])


def retarget_csv(g1_path: Path, t800_path: Path) -> None:
  """单个 G1 CSV → T800 CSV。"""
  data = np.loadtxt(g1_path, delimiter=",")
  out = retarget_array(data)
  np.savetxt(t800_path, out, delimiter=",", fmt="%.8f")


def main() -> None:
  parser = argparse.ArgumentParser(description="G1 → T800 动作数据重定向")
  parser.add_argument("--input-dir", type=str, default="datasets/csv/forward")
  parser.add_argument("--output-dir", type=str, default="datasets/csv/t800")
  args = parser.parse_args()

  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  g1_files = sorted(input_dir.glob("*.csv"))
  if not g1_files:
    print(f"未找到 CSV: {input_dir}")
    return

  for g1_file in g1_files:
    t800_file = output_dir / g1_file.name.replace("g1", "t800")
    retarget_csv(g1_file, t800_file)
    print(f"  {g1_file.name} → {t800_file.name}")

  print(f"完成：{len(g1_files)} 个文件 → {output_dir}")


if __name__ == "__main__":
  main()
