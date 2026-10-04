"""Regression for the walk12 EE semantics recorded in the original NPZ files."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from prior_contract import apply_walk12_prior_contract


class PriorContractTest(unittest.TestCase):
    def test_walk12_uses_original_base_and_elbows_without_changing_physics(self):
        params = {"num_joints": 25, "ee_body_names": ("wrong",), "gsi_buffer_size": 4096}
        rewards, sensors = object(), object()
        cfg = SimpleNamespace(events={"init_smp_state": SimpleNamespace(params=params)},
                              rewards=rewards, sensors=sensors)
        apply_walk12_prior_contract(cfg)
        self.assertEqual(params["ee_body_names"], (
            "LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R", "LINK_BASE",
            "LINK_ELBOW_YAW_L", "LINK_ELBOW_YAW_R"))
        self.assertEqual(params["gsi_buffer_size"], 4096)
        self.assertIs(cfg.rewards, rewards)
        self.assertIs(cfg.sensors, sensors)

    def test_rejects_other_joint_layout(self):
        cfg = SimpleNamespace(events={"init_smp_state": SimpleNamespace(params={"num_joints": 29})})
        with self.assertRaises(ValueError):
            apply_walk12_prior_contract(cfg)


if __name__ == "__main__":
    unittest.main()
