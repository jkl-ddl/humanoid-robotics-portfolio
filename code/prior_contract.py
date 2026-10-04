"""The colleague's walk12 checkpoint uses base/elbow EE positions, not wrists.

Names are taken from the original csv_to_npz.py, feature_to_state.py and all
twelve NPZ ee_body_names records. Physical foot sensors remain unchanged.
"""

WALK12_EE_BODY_NAMES = (
    "LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R", "LINK_BASE",
    "LINK_ELBOW_YAW_L", "LINK_ELBOW_YAW_R",
)


def apply_walk12_prior_contract(cfg):
    event = cfg.events["init_smp_state"]
    if event.params["num_joints"] != 25:
        raise ValueError("walk12 prior requires the 25-joint T800 feature contract")
    event.params["ee_body_names"] = WALK12_EE_BODY_NAMES
