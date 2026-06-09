"""Embodiment configuration: everything env code needs to know about a robot.

A ManipulationEnv is fully determined by a RobotConfig + a Task. Adding a new
arm/hand combo means writing one of these (plus an MJCF scene), no env changes.
"""

from dataclasses import dataclass
from typing import Callable

from etils import epath


@dataclass(frozen=True)
class RobotConfig:
    """Static description of a robot embodiment within a scene MJCF."""

    name: str
    # Scene MJCF containing robot + table/floor (+ task objects for now).
    scene_xml: epath.Path
    # Extra assets (meshes, included xmls) needed to compile scene_xml.
    assets: Callable[[], dict[str, bytes]]

    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    # Joint targeted by each actuator, in actuator order (length nu). For
    # coupled grippers list the driven joint once (e.g. panda: 7 arm joints +
    # finger_joint1 for the single tendon actuator).
    ctrl_joints: tuple[str, ...]

    # End-effector site used for reach rewards / goal frames.
    ee_site: str
    home_keyframe: str

    # Contact "found" sensors that fire when the hand touches the floor
    # (defined in the scene xml); used for collision penalties.
    floor_contact_sensors: tuple[str, ...] = ()

    @property
    def joints(self) -> tuple[str, ...]:
        return self.arm_joints + self.gripper_joints
