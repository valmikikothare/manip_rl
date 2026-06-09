"""Franka Panda + parallel gripper, reusing MuJoCo Playground's scene assets."""

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda

from manip_rl.robots.base import RobotConfig

_XML_DIR = mjx_env.ROOT_PATH / "manipulation" / "franka_emika_panda" / "xmls"

PANDA = RobotConfig(
    name="panda",
    scene_xml=_XML_DIR / "mjx_single_cube.xml",
    assets=panda.get_assets,
    arm_joints=(
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ),
    gripper_joints=("finger_joint1", "finger_joint2"),
    # 7 arm actuators + one tendon-coupled gripper actuator driving finger_joint1.
    ctrl_joints=(
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
        "finger_joint1",
    ),
    ee_site="gripper",
    home_keyframe="home",
    floor_contact_sensors=(
        "left_finger_pad_floor_found",
        "right_finger_pad_floor_found",
        "hand_capsule_floor_found",
    ),
)
