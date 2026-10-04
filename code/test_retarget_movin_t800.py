"""Regression for the metre-based MOVIN -> centimetre-based GMR loader bridge."""
import unittest

import numpy as np

from retarget_movin_t800 import adapt_movin_frames


class UnitBridgeTest(unittest.TestCase):
    def test_metres_and_shared_foot_alias_are_converted_once(self):
        foot = np.array([.002, 0., .001])
        identity = np.array([1., 0., 0., 0.])
        frame = {
            "Hips": [np.array([.001, 0., .01]), identity],
            "Spine1": [np.array([0., 0., .013]), identity],
            "LeftFoot": [foot, identity],
            "LeftFootMod": [foot, identity],
        }
        result = adapt_movin_frames([frame])[0]
        np.testing.assert_allclose(result["Hips"][0], [.1, 0., 1.])
        np.testing.assert_allclose(result["LeftFoot"][0], [.2, 0., .1])
        np.testing.assert_allclose(result["LeftFootMod"][0], [.2, 0., .1])
        np.testing.assert_allclose(result["Spine2"][0], [0., 0., 1.3])
        np.testing.assert_allclose(foot, [.002, 0., .001])


if __name__ == "__main__":
    unittest.main()
