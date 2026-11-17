#!/usr/bin/env python3
"""Interactive CARLA 0.9.16 Prius/mannequin eye-point calibration.

This tool places ``walker.pedestrian.0022`` relative to
``vehicle.toyota.prius``, reads the real walker eye/head/pelvis bones, and
reports their Toyota-Prius-local coordinates.  The default frozen standing
pose is useful only for visual geometry calibration.  It is not a validated
automotive seated posture or a Korean-female anthropometric model.

CARLA axes used throughout this file are +X forward, +Y right, +Z up.  Thus a
left-hand-drive root normally has a negative local Y coordinate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import carla
except ImportError as exc:  # pragma: no cover - depends on local installation
    carla = None  # type: ignore[assignment]
    CARLA_IMPORT_ERROR: Optional[BaseException] = exc
else:
    CARLA_IMPORT_ERROR = None

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


VEHICLE_BLUEPRINT = "vehicle.toyota.prius"
WALKER_BLUEPRINT = "walker.pedestrian.0022"
DEFAULT_REFERENCE_CSV = Path(
    "/home/tukorea31/carla-0.9.16/carla_female_walker_measurements.csv"
)
DEFAULT_JSON_OUTPUT = Path("/home/tukorea31/prius_driver_mannequin_calibration.json")
DEFAULT_CSV_OUTPUT = Path("/home/tukorea31/prius_driver_eye_measurement.csv")
DEFAULT_BONE_MAPPING_OUTPUT = Path(
    "/home/tukorea31/walker_0022_driver_pose_bone_mapping.txt"
)
DEFAULT_AXIS_MAP_OUTPUT = Path("/home/tukorea31/walker_0022_bone_axis_map.json")
DEFAULT_VALIDATED_POSE_OUTPUT = Path(
    "/home/tukorea31/validated_driver_seated_pose.json"
)
DEFAULT_SEATED_CSV_OUTPUT = Path(
    "/home/tukorea31/prius_seated_driver_eye_measurement.csv"
)

NEUTRAL_STANDING_EYE_PRIUS_LOCAL = (0.2739, -0.3378, 1.3885)
DEMONSTRATION_CAMERA_PRIUS_LOCAL = (0.69, -0.36, 1.20)
TRUNK_TARGET_DEG = 23.8
TRUNK_VALIDATION_TOLERANCE_DEG = 3.0
NECK_TARGET_DEG = -3.1
NECK_VALIDATION_TOLERANCE_DEG = 3.0
POSTURE_MEASUREMENT_DEFINITION_VERSION = 2

LITERATURE_TARGETS: Mapping[str, Optional[float]] = {
    "trunk_angle_deg": TRUNK_TARGET_DEG,
    "left_hip_angle_deg": 98.2,
    "right_hip_angle_deg": 103.8,
    "left_knee_angle_deg": 109.3,
    "right_knee_angle_deg": 120.4,
    "left_ankle_angle_deg": 96.4,
    "right_ankle_angle_deg": 95.5,
    "neck_segment_angle_deg": NECK_TARGET_DEG,
    "pelvis_rearward_change_from_standing_deg": 55.4,
    "head_angle_deg": None,
}

LITERATURE_POSE_NOTICE = """IMPORTANT

This seated-driver pose is based on published automotive
driver-posture literature and CARLA skeleton mapping.

walker.pedestrian.0022 is NOT a validated Korean female P50
anthropometric mannequin.

The resulting eye position is therefore a CARLA
literature-based driver-eye reference.

Korean female anthropometric validation requires a separate
comparison/correction using Size Korea or another validated
anthropometric dataset."""

IMPORTANT_NOTICE = """IMPORTANT

walker.pedestrian.0022 was selected as a representative
CARLA adult-female geometry candidate.

It is NOT currently validated as a Korean female P50
anthropometric mannequin.

The current calibration establishes the mannequin eye
location relative to the Toyota Prius geometry.

A validated seated posture and anthropometric reference
are required before claiming a Korean female representative
driver eye point."""

CSV_FIELDS = [
    "vehicle_id",
    "walker_id",
    "pose_validated",
    "walker_root_local_x",
    "walker_root_local_y",
    "walker_root_local_z",
    "walker_root_local_yaw",
    "pelvis_local_x",
    "pelvis_local_y",
    "pelvis_local_z",
    "head_local_x",
    "head_local_y",
    "head_local_z",
    "left_eye_local_x",
    "left_eye_local_y",
    "left_eye_local_z",
    "right_eye_local_x",
    "right_eye_local_y",
    "right_eye_local_z",
    "eye_mid_local_x",
    "eye_mid_local_y",
    "eye_mid_local_z",
    "compare_camera_x",
    "compare_camera_y",
    "compare_camera_z",
    "camera_eye_dx",
    "camera_eye_dy",
    "camera_eye_dz",
    "camera_eye_distance_m",
]

SEATED_DRIVER_CSV_FIELDS = [
    "walker_id",
    "pose_source",
    "literature_pose_validated",
    "korean_female_validated",
    "pelvis_local_x", "pelvis_local_y", "pelvis_local_z",
    "head_local_x", "head_local_y", "head_local_z",
    "left_eye_local_x", "left_eye_local_y", "left_eye_local_z",
    "right_eye_local_x", "right_eye_local_y", "right_eye_local_z",
    "eye_mid_local_x", "eye_mid_local_y", "eye_mid_local_z",
    "trunk_angle_deg", "hip_angle_deg", "knee_angle_deg",
    "ankle_angle_deg", "head_angle_deg",
    "demo_camera_x", "demo_camera_y", "demo_camera_z",
    "demo_eye_dx", "demo_eye_dy", "demo_eye_dz", "demo_eye_distance_m",
]


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    @classmethod
    def from_carla(cls, point: Any) -> "Point3":
        return cls(float(point.x), float(point.y), float(point.z))

    def to_location(self) -> Any:
        return carla.Location(x=self.x, y=self.y, z=self.z)

    def minus(self, other: "Point3") -> "Point3":
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance_to(self, other: "Point3") -> float:
        delta = self.minus(other)
        return math.sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)

    def midpoint(self, other: "Point3") -> "Point3":
        return Point3(
            (self.x + other.x) * 0.5,
            (self.y + other.y) * 0.5,
            (self.z + other.z) * 0.5,
        )

    def as_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class WalkerRootLocal:
    x: float
    y: float
    z: float
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    def to_transform(self) -> Any:
        return carla.Transform(
            carla.Location(x=self.x, y=self.y, z=self.z),
            carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll),
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
        }


@dataclass
class BoneRecord:
    name: str
    world: Point3


@dataclass
class FullBoneRecord:
    name: str
    world_transform: Any
    component_transform: Any
    relative_transform: Any


@dataclass
class AnatomicalAngles:
    trunk_angle_deg: Optional[float] = None
    left_hip_angle_deg: Optional[float] = None
    right_hip_angle_deg: Optional[float] = None
    left_knee_angle_deg: Optional[float] = None
    right_knee_angle_deg: Optional[float] = None
    left_ankle_angle_deg: Optional[float] = None
    right_ankle_angle_deg: Optional[float] = None
    neck_segment_angle_deg: Optional[float] = None
    head_angle_deg: Optional[float] = None

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class TrunkPostureGeometry:
    """Final landmark geometry measured in Prius-local coordinates.

    Positive inclination means that the upper landmark is rearward (-X) of
    the lower landmark; negative inclination means forward (+X).  The angle
    magnitude is measured from the vehicle +Z axis and includes any lateral
    offset, while the sign comes only from the longitudinal direction.
    """

    pelvis_local: Point3
    shoulder_mid_local: Point3
    neck_local: Point3
    head_local: Point3
    trunk_inclination_deg: float
    neck_inclination_deg: float


@dataclass
class SkeletonSnapshot:
    world: Dict[str, Point3]
    vehicle_local: Dict[str, Point3]
    bone_names: Dict[str, str]


@dataclass(frozen=True)
class PoseBone:
    name: str
    location_m: Tuple[float, float, float]
    rotation_deg: Tuple[float, float, float]


@dataclass
class SeatedPoseSpec:
    metadata: Dict[str, Any]
    bones: List[PoseBone]
    path: Path

    @property
    def validated(self) -> bool:
        literature_scope = (
            self.metadata.get("literature_pose_validated") is True
            and self.metadata.get("carla_bone_axis_validated") is True
            and self.metadata.get("seat_reference_calibrated") is True
        )
        legacy_scope = (
            self.metadata.get("anthropometrically_valid") is True
            and self.metadata.get("seat_reference_calibrated") is True
        )
        return literature_scope or legacy_scope


@dataclass
class RuntimeState:
    root: WalkerRootLocal
    pose_validated: bool
    pose_description: str
    snapshot: SkeletonSnapshot
    compare_camera: Optional[Point3]
    anatomical_angles: Optional[AnatomicalAngles] = None
    literature_pose_validated: bool = False
    pose_stage: str = "neutral"
    seat_target: Optional[Point3] = None
    trunk_posture: Optional[TrunkPostureGeometry] = None
    saved_once: bool = False


class CameraFrameBuffer:
    """Thread-safe copy of the latest BGRA CARLA camera image."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[Tuple[bytes, int, int]] = None

    def callback(self, image: Any) -> None:
        with self._lock:
            self._frame = (bytes(image.raw_data), int(image.width), int(image.height))

    def pygame_surface(self) -> Optional[Any]:
        if pygame is None or np is None:
            return None
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        raw, width, height = frame
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
        rgb = bgra[:, :, :3][:, :, ::-1]
        return pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))


EXACT_BONES: Mapping[str, Sequence[str]] = {
    "left_eye": ("crl_eye__L", "eye_l", "left_eye", "eye_left", "LeftEye"),
    "right_eye": ("crl_eye__R", "eye_r", "right_eye", "eye_right", "RightEye"),
    "head": ("crl_Head__C", "head", "Head", "head_c", "b_head"),
    "pelvis": ("crl_hips__C", "pelvis", "Pelvis", "hips", "Hips", "hip"),
}

DRIVER_BONE_EXACT: Mapping[str, Sequence[str]] = {
    "pelvis": ("crl_hips__C", "pelvis", "hips"),
    "lower_spine": ("crl_spine__C", "spine", "spine_01"),
    "upper_spine": ("crl_spine01__C", "spine01", "chest", "upper_spine"),
    "neck": ("crl_neck__C", "neck"),
    "head": ("crl_Head__C", "head"),
    "left_eye": ("crl_eye__L", "left_eye", "eye_l"),
    "right_eye": ("crl_eye__R", "right_eye", "eye_r"),
    "left_upper_leg": ("crl_thigh__L", "left_thigh", "thigh_l", "upperleg_l"),
    "right_upper_leg": ("crl_thigh__R", "right_thigh", "thigh_r", "upperleg_r"),
    "left_lower_leg": ("crl_leg__L", "left_leg", "calf_l", "lowerleg_l"),
    "right_lower_leg": ("crl_leg__R", "right_leg", "calf_r", "lowerleg_r"),
    "left_foot": ("crl_foot__L", "left_foot", "foot_l"),
    "right_foot": ("crl_foot__R", "right_foot", "foot_r"),
    "left_toe": ("crl_toe__L", "left_toe", "toe_l", "ball_l"),
    "right_toe": ("crl_toe__R", "right_toe", "toe_r", "ball_r"),
    "left_clavicle": ("crl_shoulder__L", "left_shoulder", "clavicle_l"),
    "right_clavicle": ("crl_shoulder__R", "right_shoulder", "clavicle_r"),
    "left_upper_arm": ("crl_arm__L", "left_arm", "upperarm_l"),
    "right_upper_arm": ("crl_arm__R", "right_arm", "upperarm_r"),
    "left_forearm": ("crl_foreArm__L", "left_forearm", "forearm_l"),
    "right_forearm": ("crl_foreArm__R", "right_forearm", "forearm_r"),
    "left_hand": ("crl_hand__L", "left_hand", "hand_l"),
    "right_hand": ("crl_hand__R", "right_hand", "hand_r"),
}

EXPECTED_DRIVER_HIERARCHY: Sequence[Tuple[str, str]] = (
    ("crl_root", "crl_hips__C"),
    ("crl_hips__C", "crl_spine__C"),
    ("crl_spine__C", "crl_spine01__C"),
    ("crl_spine01__C", "crl_neck__C"),
    ("crl_neck__C", "crl_Head__C"),
    ("crl_Head__C", "crl_eye__L"),
    ("crl_Head__C", "crl_eye__R"),
    ("crl_hips__C", "crl_thigh__L"),
    ("crl_thigh__L", "crl_leg__L"),
    ("crl_leg__L", "crl_foot__L"),
    ("crl_foot__L", "crl_toe__L"),
    ("crl_hips__C", "crl_thigh__R"),
    ("crl_thigh__R", "crl_leg__R"),
    ("crl_leg__R", "crl_foot__R"),
    ("crl_foot__R", "crl_toe__R"),
    ("crl_spine01__C", "crl_shoulder__L"),
    ("crl_shoulder__L", "crl_arm__L"),
    ("crl_arm__L", "crl_foreArm__L"),
    ("crl_foreArm__L", "crl_hand__L"),
    ("crl_spine01__C", "crl_shoulder__R"),
    ("crl_shoulder__R", "crl_arm__R"),
    ("crl_arm__R", "crl_foreArm__R"),
    ("crl_foreArm__R", "crl_hand__R"),
)

POSE_CONTROL_TO_ANATOMICAL_ANGLE: Mapping[str, str] = {
    "lower_spine": "trunk_angle_deg",
    "neck": "neck_segment_angle_deg",
    "left_upper_leg": "left_hip_angle_deg",
    "right_upper_leg": "right_hip_angle_deg",
    "left_lower_leg": "left_knee_angle_deg",
    "right_lower_leg": "right_knee_angle_deg",
    "left_foot": "left_ankle_angle_deg",
    "right_foot": "right_ankle_angle_deg",
}


def _normalise_bone_name(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")


def _side_markers(name: str) -> Tuple[bool, bool]:
    tokens = [token for token in name.split("_") if token]
    left = "left" in tokens or "l" in tokens or name.startswith(("left", "leye"))
    right = "right" in tokens or "r" in tokens or name.startswith(("right", "reye"))
    return left, right


def _bone_score(name: str, role: str) -> Optional[int]:
    normalised = _normalise_bone_name(name)
    exact = [_normalise_bone_name(candidate) for candidate in EXACT_BONES[role]]
    if normalised in exact:
        return 10_000 - exact.index(normalised)
    if role in ("left_eye", "right_eye"):
        if "eye" not in normalised or "brow" in normalised or "lid" in normalised:
            return None
        left, right = _side_markers(normalised)
        expected_left = role == "left_eye"
        if expected_left and (not left or right):
            return None
        if not expected_left and (not right or left):
            return None
        penalty = 100 if normalised.endswith("_a") or "aux" in normalised else 0
        return 500 - penalty - len(normalised)
    if role == "head" and "head" in normalised:
        penalty = 100 if any(x in normalised for x in ("top", "end", "socket")) else 0
        return 450 - penalty - len(normalised)
    if role == "pelvis" and any(x in normalised for x in ("pelvis", "hips", "hip")):
        left, right = _side_markers(normalised)
        return 475 - (150 if left or right else 0) - len(normalised)
    return None


def find_skeleton_bones(records: Sequence[BoneRecord]) -> Dict[str, BoneRecord]:
    selected: Dict[str, BoneRecord] = {}
    for role in EXACT_BONES:
        matches: List[Tuple[int, str, BoneRecord]] = []
        for record in records:
            score = _bone_score(record.name, role)
            if score is not None:
                matches.append((score, record.name.casefold(), record))
        if matches:
            matches.sort(key=lambda item: (-item[0], item[1]))
            selected[role] = matches[0][2]
    if selected.get("left_eye") is selected.get("right_eye"):
        selected.pop("right_eye", None)
    return selected


def _driver_bone_score(name: str, role: str) -> Optional[int]:
    normalised = _normalise_bone_name(name)
    candidates = [_normalise_bone_name(item) for item in DRIVER_BONE_EXACT[role]]
    if normalised in candidates:
        return 10_000 - candidates.index(normalised)
    left, right = _side_markers(normalised)
    if role.startswith("left_") and (not left or right):
        return None
    if role.startswith("right_") and (not right or left):
        return None
    keyword_sets: Mapping[str, Sequence[str]] = {
        "pelvis": ("pelvis", "hips", "hip"),
        "lower_spine": ("spine", "lumbar"),
        "upper_spine": ("chest", "thorax", "spine01", "spine_01"),
        "neck": ("neck",),
        "head": ("head",),
        "left_eye": ("eye",),
        "right_eye": ("eye",),
        "left_upper_leg": ("thigh", "upperleg", "upper_leg"),
        "right_upper_leg": ("thigh", "upperleg", "upper_leg"),
        "left_lower_leg": ("calf", "lowerleg", "lower_leg", "leg"),
        "right_lower_leg": ("calf", "lowerleg", "lower_leg", "leg"),
        "left_foot": ("foot", "ankle"),
        "right_foot": ("foot", "ankle"),
        "left_toe": ("toe", "ball"),
        "right_toe": ("toe", "ball"),
        "left_clavicle": ("clavicle", "shoulder"),
        "right_clavicle": ("clavicle", "shoulder"),
        "left_upper_arm": ("upperarm", "upper_arm", "arm"),
        "right_upper_arm": ("upperarm", "upper_arm", "arm"),
        "left_forearm": ("forearm", "fore_arm"),
        "right_forearm": ("forearm", "fore_arm"),
        "left_hand": ("hand", "wrist"),
        "right_hand": ("hand", "wrist"),
    }
    if not any(keyword in normalised for keyword in keyword_sets[role]):
        return None
    penalties = ("end", "twist", "aux", "ik", "index", "thumb", "middle", "ring", "pinky")
    return 500 - sum(120 for token in penalties if token in normalised) - len(normalised)


def map_driver_bones(records: Sequence[FullBoneRecord]) -> Dict[str, FullBoneRecord]:
    selected: Dict[str, FullBoneRecord] = {}
    for role in DRIVER_BONE_EXACT:
        matches: List[Tuple[int, str, FullBoneRecord]] = []
        for record in records:
            score = _driver_bone_score(record.name, role)
            if score is not None:
                matches.append((score, record.name.casefold(), record))
        if matches:
            matches.sort(key=lambda item: (-item[0], item[1]))
            selected[role] = matches[0][2]
    return selected


def read_full_bones(walker: Any, world: Any, args: argparse.Namespace) -> List[FullBoneRecord]:
    last_error: Optional[BaseException] = None
    for _ in range(args.bone_retries):
        try:
            output = walker.get_bones()
            records = [
                FullBoneRecord(
                    name=str(item.name),
                    world_transform=_copy_transform(item.world),
                    component_transform=_copy_transform(item.component),
                    relative_transform=_copy_transform(item.relative),
                )
                for item in output.bone_transforms
            ]
            if records:
                return records
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            last_error = exc
        advance_world(world, 1, args.timeout)
    if last_error is not None:
        raise RuntimeError(f"get_bones() full-transform read failed: {last_error}")
    raise RuntimeError("get_bones() repeatedly returned no full bone transforms")


def _format_transform(transform: Any) -> str:
    location = transform.location
    rotation = transform.rotation
    return (
        f"loc=({location.x:.9f}, {location.y:.9f}, {location.z:.9f}) m "
        f"rot=(pitch={rotation.pitch:.9f}, yaw={rotation.yaw:.9f}, "
        f"roll={rotation.roll:.9f}) deg"
    )


def dump_driver_bone_mapping(
    records: Sequence[FullBoneRecord], output_path: Path
) -> Dict[str, FullBoneRecord]:
    mapping = map_driver_bones(records)
    names = {record.name for record in records}
    missing_roles = [role for role in DRIVER_BONE_EXACT if role not in mapping]
    missing_edges = [
        (parent, child)
        for parent, child in EXPECTED_DRIVER_HIERARCHY
        if parent not in names or child not in names
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(f"walker_id: {WALKER_BLUEPRINT}\n")
        stream.write("status: LIVE get_bones() DUMP\n")
        stream.write(f"bone_count: {len(records)}\n")
        stream.write("spaces: world, component(actor), relative(parent bone)\n\n")
        stream.write("API NOTE: WalkerBoneControlOut does not expose parent names.\n")
        stream.write("Hierarchy edges below come from CARLA 0.9.16 skeleton.txt and\n")
        stream.write("are validated by live endpoint-name presence.\n\n")
        stream.write("ANATOMICAL MAPPING\n")
        for role in DRIVER_BONE_EXACT:
            record = mapping.get(role)
            stream.write(f"{role}: {record.name if record else 'NOT FOUND'}\n")
        stream.write("\nEXPECTED HIERARCHY ENDPOINT CHECK\n")
        for parent, child in EXPECTED_DRIVER_HIERARCHY:
            status = "PASS" if parent in names and child in names else "FAIL"
            stream.write(f"{status}: {parent} -> {child}\n")
        stream.write("\nFULL LIVE BONE TRANSFORMS\n")
        for index, record in enumerate(records):
            stream.write(f"[{index:03d}] {record.name}\n")
            stream.write(f"  world:    {_format_transform(record.world_transform)}\n")
            stream.write(f"  component:{_format_transform(record.component_transform)}\n")
            stream.write(f"  relative: {_format_transform(record.relative_transform)}\n")
    if missing_roles:
        raise RuntimeError(
            "Driver-pose anatomical bones not found: " + ", ".join(missing_roles)
        )
    if missing_edges:
        missing = ", ".join(f"{a}->{b}" for a, b in missing_edges)
        raise RuntimeError("Live walker differs from expected CARLA hierarchy: " + missing)
    print(f"Live walker bone mapping saved: {output_path}")
    return mapping


def _point_in_vehicle(vehicle_transform: Any, transform: Any) -> Point3:
    return Point3.from_carla(
        world_to_vehicle_local(vehicle_transform, transform.location)
    )


def _included_angle_xz(vertex: Point3, first: Point3, second: Point3) -> float:
    ax, az = first.x - vertex.x, first.z - vertex.z
    bx, bz = second.x - vertex.x, second.z - vertex.z
    norm_a = math.hypot(ax, az)
    norm_b = math.hypot(bx, bz)
    if norm_a < 1.0e-8 or norm_b < 1.0e-8:
        raise ValueError("Cannot measure angle from a zero-length sagittal segment")
    cosine = max(-1.0, min(1.0, (ax * bx + az * bz) / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def _signed_inclination_from_vehicle_vertical(
    lower: Point3, upper: Point3
) -> float:
    """Return landmark-vector inclination from Prius +Z, not a bone Euler angle."""

    dx = upper.x - lower.x
    dy = upper.y - lower.y
    dz = upper.z - lower.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1.0e-8:
        raise ValueError("Cannot measure inclination from coincident landmarks")
    unsigned = math.degrees(math.atan2(math.hypot(dx, dy), dz))
    if abs(dx) < 1.0e-8:
        return unsigned
    # CARLA/Prius +X is forward. An upper point behind the lower point is a
    # positive rearward driver-posture inclination.
    return unsigned if dx < 0.0 else -unsigned


def measure_trunk_posture_geometry(
    vehicle: Any, records: Sequence[FullBoneRecord]
) -> TrunkPostureGeometry:
    """Measure pelvis-to-shoulder and neck-to-head vectors from get_bones()."""

    mapping = map_driver_bones(records)
    required = ("pelvis", "left_clavicle", "right_clavicle", "neck", "head")
    missing = [role for role in required if role not in mapping]
    if missing:
        raise RuntimeError(
            "Cannot measure trunk/neck landmark geometry; missing " + ", ".join(missing)
        )
    vehicle_transform = vehicle.get_transform()
    points = {
        role: _point_in_vehicle(vehicle_transform, mapping[role].world_transform)
        for role in required
    }
    shoulder_mid = points["left_clavicle"].midpoint(points["right_clavicle"])
    return TrunkPostureGeometry(
        pelvis_local=points["pelvis"],
        shoulder_mid_local=shoulder_mid,
        neck_local=points["neck"],
        head_local=points["head"],
        trunk_inclination_deg=_signed_inclination_from_vehicle_vertical(
            points["pelvis"], shoulder_mid
        ),
        neck_inclination_deg=_signed_inclination_from_vehicle_vertical(
            points["neck"], points["head"]
        ),
    )


def measure_anatomical_angles(
    vehicle: Any, records: Sequence[FullBoneRecord]
) -> AnatomicalAngles:
    mapping = map_driver_bones(records)
    required = (
        "left_clavicle", "right_clavicle", "left_upper_leg", "right_upper_leg",
        "left_lower_leg", "right_lower_leg", "left_foot", "right_foot",
        "left_toe", "right_toe", "neck", "head", "pelvis",
    )
    missing = [role for role in required if role not in mapping]
    if missing:
        raise RuntimeError("Cannot measure driver angles; missing " + ", ".join(missing))
    transform = vehicle.get_transform()
    points = {
        role: _point_in_vehicle(transform, record.world_transform)
        for role, record in mapping.items()
    }
    posture = measure_trunk_posture_geometry(vehicle, records)
    return AnatomicalAngles(
        trunk_angle_deg=posture.trunk_inclination_deg,
        left_hip_angle_deg=_included_angle_xz(
            points["left_upper_leg"], points["left_clavicle"], points["left_lower_leg"]
        ),
        right_hip_angle_deg=_included_angle_xz(
            points["right_upper_leg"], points["right_clavicle"], points["right_lower_leg"]
        ),
        left_knee_angle_deg=_included_angle_xz(
            points["left_lower_leg"], points["left_upper_leg"], points["left_foot"]
        ),
        right_knee_angle_deg=_included_angle_xz(
            points["right_lower_leg"], points["right_upper_leg"], points["right_foot"]
        ),
        left_ankle_angle_deg=_included_angle_xz(
            points["left_foot"], points["left_lower_leg"], points["left_toe"]
        ),
        right_ankle_angle_deg=_included_angle_xz(
            points["right_foot"], points["right_lower_leg"], points["right_toe"]
        ),
        neck_segment_angle_deg=posture.neck_inclination_deg,
        head_angle_deg=None,
    )


def vehicle_local_to_world(vehicle_transform: Any, local_location: Any) -> Any:
    """Transform a point with the complete CARLA vehicle transform."""

    point = carla.Location(
        x=float(local_location.x),
        y=float(local_location.y),
        z=float(local_location.z),
    )
    vehicle_transform.transform(point)
    return point


def world_to_vehicle_local(vehicle_transform: Any, world_location: Any) -> Any:
    """Inverse-transform a world point, including vehicle rotation."""

    point = carla.Location(
        x=float(world_location.x),
        y=float(world_location.y),
        z=float(world_location.z),
    )
    vehicle_transform.inverse_transform(point)
    return point


def _multiply_rotation_matrices(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    return [
        [sum(float(a[row][k]) * float(b[k][col]) for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _rotation_from_matrix(matrix: Sequence[Sequence[float]]) -> Any:
    """Extract CARLA Rz(yaw)*Ry(pitch)*Rx(roll) Euler angles."""

    pitch = math.asin(max(-1.0, min(1.0, float(matrix[2][0]))))
    cp = math.cos(pitch)
    if abs(cp) > 1.0e-7:
        yaw = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
        roll = math.atan2(-float(matrix[2][1]), float(matrix[2][2]))
    else:
        yaw = math.atan2(-float(matrix[0][1]), float(matrix[1][1]))
        roll = 0.0
    return carla.Rotation(
        pitch=math.degrees(pitch),
        yaw=math.degrees(yaw),
        roll=math.degrees(roll),
    )


def vehicle_local_transform_to_world(vehicle_transform: Any, local_transform: Any) -> Any:
    """Compose complete parent/child transforms, not just translations."""

    world_location = vehicle_local_to_world(vehicle_transform, local_transform.location)
    vehicle_matrix = vehicle_transform.get_matrix()
    local_matrix = local_transform.get_matrix()
    rotation_matrix = _multiply_rotation_matrices(
        [row[:3] for row in vehicle_matrix[:3]],
        [row[:3] for row in local_matrix[:3]],
    )
    return carla.Transform(world_location, _rotation_from_matrix(rotation_matrix))


def apply_anthropometric_eye_correction(snapshot: SkeletonSnapshot) -> SkeletonSnapshot:
    """Reserved extension point; no unvalidated correction exists in this version."""

    raise NotImplementedError(
        "No anthropometric eye correction is defined. Do not place the eye 0.755 m "
        "above a pelvis bone: seated eye height uses a seat-surface reference."
    )


def advance_world(world: Any, count: int, timeout: float) -> None:
    for _ in range(max(0, count)):
        if world.get_settings().synchronous_mode:
            world.tick(timeout)
        else:
            world.wait_for_tick(timeout)


def _copy_transform(transform: Any) -> Any:
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
        ),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll),
        ),
    )


def spawn_prius(world: Any, args: argparse.Namespace) -> Tuple[Any, Any]:
    library = world.get_blueprint_library()
    try:
        blueprint = library.find(VEHICLE_BLUEPRINT)
    except RuntimeError as exc:
        raise RuntimeError(f"Missing required blueprint {VEHICLE_BLUEPRINT}: {exc}") from exc
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("The current map has no vehicle spawn points")
    start = args.spawn_index % len(spawn_points)
    candidates = spawn_points[start:] + spawn_points[:start]
    vehicle = None
    selected_transform = None
    for candidate in candidates[: args.max_spawn_attempts]:
        selected_transform = _copy_transform(candidate)
        vehicle = world.try_spawn_actor(blueprint, selected_transform)
        if vehicle is not None:
            break
    if vehicle is None or selected_transform is None:
        raise RuntimeError("Could not spawn vehicle.toyota.prius at a clear map spawn point")
    try:
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )
        advance_world(world, 3, args.timeout)
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_simulate_physics(False)
        vehicle.set_enable_gravity(False)
        vehicle.set_transform(selected_transform)
        advance_world(world, 2, args.timeout)
        return vehicle, selected_transform
    except BaseException:
        if vehicle.is_alive:
            vehicle.destroy()
        raise


def _walker_safe_spawn_candidates(vehicle_transform: Any) -> List[Any]:
    local_candidates = (
        carla.Location(x=0.0, y=-3.5, z=1.0),
        carla.Location(x=3.5, y=0.0, z=1.0),
        carla.Location(x=-3.5, y=0.0, z=1.0),
        carla.Location(x=0.0, y=3.5, z=1.0),
    )
    return [
        carla.Transform(
            vehicle_local_to_world(vehicle_transform, location),
            carla.Rotation(yaw=float(vehicle_transform.rotation.yaw)),
        )
        for location in local_candidates
    ]


def spawn_walker_near_prius(world: Any, vehicle: Any, args: argparse.Namespace) -> Any:
    library = world.get_blueprint_library()
    try:
        blueprint = library.find(WALKER_BLUEPRINT)
    except RuntimeError as exc:
        raise RuntimeError(f"Missing required blueprint {WALKER_BLUEPRINT}: {exc}") from exc
    walker = None
    for candidate in _walker_safe_spawn_candidates(vehicle.get_transform()):
        walker = world.try_spawn_actor(blueprint, candidate)
        if walker is not None:
            break
    if walker is None:
        raise RuntimeError("Could not spawn walker.pedestrian.0022 near the Prius")
    try:
        walker.apply_control(carla.WalkerControl())
        advance_world(world, 3, args.timeout)
        walker.get_pose_from_animation()
        walker.show_pose()
        advance_world(world, 2, args.timeout)
        # These methods are present on carla.Actor/Walker in the installed
        # 0.9.16 binding. Geometry clipping is inspected visually, not solved
        # or certified by collision response.
        walker.set_collisions(False)
        walker.set_enable_gravity(False)
        walker.set_simulate_physics(False)
        advance_world(world, 1, args.timeout)
        return walker
    except BaseException:
        if walker.is_alive:
            walker.destroy()
        raise


def _float_triplet(value: Any, name: str) -> Tuple[float, float, float]:
    if isinstance(value, Mapping):
        xyz = ("x", "y", "z")
        pyr = ("pitch", "yaw", "roll")
        if all(key in value for key in xyz):
            return tuple(float(value[key]) for key in xyz)  # type: ignore[return-value]
        if all(key in value for key in pyr):
            return tuple(float(value[key]) for key in pyr)  # type: ignore[return-value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3:
        return float(value[0]), float(value[1]), float(value[2])
    raise ValueError(f"{name} must contain three numeric values")


def load_validated_seated_pose(path: Path) -> SeatedPoseSpec:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("seated pose JSON root must be an object")
    metadata = payload.get("metadata", {})
    raw_bones = payload.get("bones")
    if not isinstance(metadata, Mapping):
        raise ValueError("seated pose metadata must be an object")
    if not isinstance(raw_bones, list) or not raw_bones:
        raise ValueError("seated pose bones must be a non-empty list")
    merged_metadata = dict(metadata)
    for key in (
        "pose_name",
        "pose_stage",
        "seat_target_vehicle_local",
        "joint_targets",
        "anthropometric_population",
    ):
        if key in payload:
            merged_metadata[key] = payload[key]
    result = SeatedPoseSpec(merged_metadata, [], path)
    if not result.validated:
        raise ValueError(
            "The external pose must either declare the legacy anthropometric/seat "
            "validation pair or metadata.literature_pose_validated=true, "
            "metadata.carla_bone_axis_validated=true, and "
            "metadata.seat_reference_calibrated=true"
        )
    seen: set[str] = set()
    for index, item in enumerate(raw_bones):
        if not isinstance(item, Mapping) or not isinstance(item.get("relative"), Mapping):
            raise ValueError(f"bones[{index}] requires name and relative")
        name = str(item.get("name", "")).strip()
        if not name or name.casefold() in seen:
            raise ValueError(f"invalid or duplicate bone name at bones[{index}]")
        seen.add(name.casefold())
        relative = item["relative"]
        location = _float_triplet(relative.get("location_m"), f"bones[{index}].location_m")
        rotation = _float_triplet(relative.get("rotation_deg"), f"bones[{index}].rotation_deg")
        if not all(math.isfinite(number) for number in (*location, *rotation)):
            raise ValueError(f"bones[{index}] contains a non-finite value")
        result.bones.append(PoseBone(name, location, rotation))
    return result


def apply_validated_seated_pose(
    walker: Any, world: Any, pose: SeatedPoseSpec, args: argparse.Namespace
) -> None:
    raw = walker.get_bones()
    available = {str(item.name).casefold(): str(item.name) for item in raw.bone_transforms}
    controls: List[Tuple[str, Any]] = []
    missing: List[str] = []
    for pose_bone in pose.bones:
        actual_name = available.get(pose_bone.name.casefold())
        if actual_name is None:
            missing.append(pose_bone.name)
            continue
        controls.append(
            (
                actual_name,
                carla.Transform(
                    carla.Location(*pose_bone.location_m),
                    carla.Rotation(*pose_bone.rotation_deg),
                ),
            )
        )
    if missing:
        raise RuntimeError("External pose bones missing from walker: " + ", ".join(missing))
    walker.get_pose_from_animation()
    walker.set_bones(carla.WalkerBoneControlIn(controls))
    walker.blend_pose(1.0)
    walker.show_pose()
    advance_world(world, args.pose_ticks, args.timeout)


def _relative_transform_with_axis_delta(
    transform: Any, axis: str, delta_deg: float
) -> Any:
    location = carla.Location(
        x=float(transform.location.x),
        y=float(transform.location.y),
        z=float(transform.location.z),
    )
    rotation = carla.Rotation(
        pitch=float(transform.rotation.pitch),
        yaw=float(transform.rotation.yaw),
        roll=float(transform.rotation.roll),
    )
    setattr(rotation, axis, getattr(rotation, axis) + delta_deg)
    return carla.Transform(location, rotation)


def _apply_parent_relative_controls(
    walker: Any, transforms: Sequence[Tuple[str, Any]]
) -> None:
    walker.set_bones(carla.WalkerBoneControlIn(list(transforms)))
    walker.blend_pose(1.0)
    walker.show_pose()


def _angle_value(angles: AnatomicalAngles, field_name: str) -> float:
    value = getattr(angles, field_name)
    if value is None:
        raise RuntimeError(f"Anatomical angle {field_name} is unavailable")
    return float(value)


def run_bone_axis_test(
    walker: Any,
    vehicle: Any,
    world: Any,
    args: argparse.Namespace,
) -> None:
    """Perturb each candidate Euler axis and record landmark-angle response.

    Automated slopes are evidence, not sufficient validation. Interactive users
    must confirm the suggested axis/sign after inspecting CARLA debug geometry.
    """

    baseline_records = read_full_bones(walker, world, args)
    mapping = dump_driver_bone_mapping(baseline_records, args.bone_mapping_output)
    baseline_angles = measure_anatomical_angles(vehicle, baseline_records)
    test_roles = tuple(POSE_CONTROL_TO_ANATOMICAL_ANGLE)
    results: Dict[str, Any] = {}
    vehicle_transform = vehicle.get_transform()
    colors = {
        "pitch": carla.Color(255, 80, 80),
        "yaw": carla.Color(80, 255, 80),
        "roll": carla.Color(80, 120, 255),
    }
    child_roles = {
        "lower_spine": "upper_spine",
        "neck": "head",
        "left_upper_leg": "left_lower_leg",
        "right_upper_leg": "right_lower_leg",
        "left_lower_leg": "left_foot",
        "right_lower_leg": "right_foot",
        "left_foot": "left_toe",
        "right_foot": "right_toe",
    }
    print("\nBONE AXIS TEST")
    print("Each local Euler axis is tested at +/- the requested step.")
    print("Red=pitch, green=yaw, blue=roll. Inspect the child-point motion.")
    for role in test_roles:
        control = mapping[role]
        child_role = child_roles[role]
        child_name = mapping[child_role].name
        angle_field = POSE_CONTROL_TO_ANATOMICAL_ANGLE[role]
        baseline_value = _angle_value(baseline_angles, angle_field)
        baseline_child = _point_in_vehicle(
            vehicle_transform, mapping[child_role].world_transform
        )
        axis_results: Dict[str, Any] = {}
        for axis in ("pitch", "yaw", "roll"):
            angle_values: Dict[str, float] = {}
            child_values: Dict[str, Dict[str, float]] = {}
            for label, delta in (("plus", args.axis_test_step), ("minus", -args.axis_test_step)):
                changed = _relative_transform_with_axis_delta(
                    control.relative_transform, axis, delta
                )
                _apply_parent_relative_controls(walker, [(control.name, changed)])
                advance_world(world, args.pose_ticks, args.timeout)
                changed_records = read_full_bones(walker, world, args)
                changed_mapping = map_driver_bones(changed_records)
                changed_angles = measure_anatomical_angles(vehicle, changed_records)
                changed_child = _point_in_vehicle(
                    vehicle.get_transform(), changed_mapping[child_role].world_transform
                )
                angle_values[label] = _angle_value(changed_angles, angle_field)
                child_values[label] = changed_child.as_dict()
                world.debug.draw_line(
                    vehicle_local_to_world(vehicle.get_transform(), baseline_child.to_location()),
                    vehicle_local_to_world(vehicle.get_transform(), changed_child.to_location()),
                    thickness=0.035,
                    color=colors[axis],
                    life_time=max(args.axis_test_hold, 1.0),
                    persistent_lines=False,
                )
                world.debug.draw_string(
                    vehicle_local_to_world(vehicle.get_transform(), changed_child.to_location()),
                    f"{role} {axis} {delta:+.0f}",
                    draw_shadow=True,
                    color=colors[axis],
                    life_time=max(args.axis_test_hold, 1.0),
                    persistent_lines=False,
                )
                if args.axis_test_hold > 0.0:
                    time.sleep(args.axis_test_hold)
            slope = (angle_values["plus"] - angle_values["minus"]) / (
                2.0 * args.axis_test_step
            )
            axis_results[axis] = {
                "plus_angle_deg": angle_values["plus"],
                "minus_angle_deg": angle_values["minus"],
                "anatomical_deg_per_carla_deg": slope,
                "plus_child_vehicle_local": child_values["plus"],
                "minus_child_vehicle_local": child_values["minus"],
            }
            _apply_parent_relative_controls(
                walker, [(control.name, _copy_transform(control.relative_transform))]
            )
            advance_world(world, args.pose_ticks, args.timeout)
        suggested_axis = max(
            axis_results,
            key=lambda item: abs(axis_results[item]["anatomical_deg_per_carla_deg"]),
        )
        suggested_slope = float(
            axis_results[suggested_axis]["anatomical_deg_per_carla_deg"]
        )
        confirmed = False
        chosen_axis: Optional[str] = None
        if not args.non_interactive:
            print(f"\n{role} ({control.name}) -> child {child_name}")
            print(f"  baseline {angle_field} = {baseline_value:.3f} deg")
            for axis in ("pitch", "yaw", "roll"):
                slope = axis_results[axis]["anatomical_deg_per_carla_deg"]
                print(f"  {axis}: anatomical/CARLA slope = {slope:+.4f}")
            answer = input(
                f"Confirm sagittal flexion axis '{suggested_axis}' after visual inspection? [y/N] "
            ).strip().casefold()
            confirmed = answer in ("y", "yes")
            chosen_axis = suggested_axis if confirmed else None
        results[role] = {
            "bone": control.name,
            "child_bone": child_name,
            "anatomical_angle": angle_field,
            "baseline_angle_deg": baseline_value,
            "tests": axis_results,
            "selected_axis": chosen_axis,
            "anatomical_deg_per_carla_deg": suggested_slope if confirmed else None,
            "human_visual_confirmation": confirmed,
            "verified": bool(confirmed and abs(suggested_slope) >= 0.5),
        }
    all_verified = all(item["verified"] for item in results.values())
    payload = {
        "walker": WALKER_BLUEPRINT,
        "carla_version": "0.9.16",
        "test_step_deg": args.axis_test_step,
        "posture_measurement_definition_version": (
            POSTURE_MEASUREMENT_DEFINITION_VERSION
        ),
        "trunk_measurement_definition": (
            "signed inclination of pelvis-to-bilateral-shoulder-midpoint vector "
            "from Prius +Z; positive is rearward"
        ),
        "neck_measurement_definition": (
            "signed inclination of neck-to-head vector from Prius +Z; "
            "negative is forward"
        ),
        "literature_pose_axis_map_validated": all_verified,
        "validation_meaning": (
            "CARLA local Euler axis/sign visually checked against measured sagittal "
            "landmark-angle response; not Korean anthropometric validation"
        ),
        "bones": results,
    }
    args.axis_map_output.parent.mkdir(parents=True, exist_ok=True)
    with args.axis_map_output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Bone-axis measurements saved: {args.axis_map_output}")
    if all_verified:
        print("CARLA BONE-AXIS VALIDATION: PASS")
    else:
        print("CARLA BONE-AXIS VALIDATION REQUIRED")


@dataclass
class PoseSolveResult:
    stage: str
    converged: bool
    iterations: int
    measured_angles: AnatomicalAngles
    controlled_relative_transforms: Dict[str, Any]
    max_error_deg: float
    baseline_angles: Optional[AnatomicalAngles] = None


def load_verified_axis_map(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping) or payload.get("walker") != WALKER_BLUEPRINT:
        raise ValueError("Bone-axis map is not for walker.pedestrian.0022")
    if payload.get("literature_pose_axis_map_validated") is not True:
        raise ValueError("CARLA BONE-AXIS VALIDATION REQUIRED")
    definition_version = payload.get("posture_measurement_definition_version")
    if definition_version != POSTURE_MEASUREMENT_DEFINITION_VERSION:
        print(
            "WARNING: axis map predates the pelvis-to-shoulder trunk definition. "
            "Its verified axis/sign will be used only as an optimizer seed; final "
            "acceptance still requires live landmark residual validation. Rerun "
            "--bone-axis-test to refresh its gain."
        )
    raw_bones = payload.get("bones")
    if not isinstance(raw_bones, Mapping):
        raise ValueError("Bone-axis map has no bones object")
    result: Dict[str, Dict[str, Any]] = {}
    for role in POSE_CONTROL_TO_ANATOMICAL_ANGLE:
        item = raw_bones.get(role)
        if not isinstance(item, Mapping) or item.get("verified") is not True:
            raise ValueError(f"Bone-axis role is not verified: {role}")
        axis = item.get("selected_axis")
        gain = item.get("anatomical_deg_per_carla_deg")
        if axis not in ("pitch", "yaw", "roll") or not isinstance(gain, (int, float)):
            raise ValueError(f"Invalid verified axis entry: {role}")
        if not math.isfinite(float(gain)) or abs(float(gain)) < 0.5:
            raise ValueError(f"Unusable anatomical/CARLA gain: {role}")
        result[role] = dict(item)
    return result


def _stage_control_roles(stage: str) -> Tuple[str, ...]:
    lower = (
        "left_upper_leg", "right_upper_leg", "left_lower_leg",
        "right_lower_leg", "left_foot", "right_foot",
    )
    if stage == "lower-body":
        return lower
    if stage in ("seated", "seated-driver"):
        return ("lower_spine", "neck") + lower
    return ()


def _validate_trunk_control_separation(roles: Sequence[str]) -> None:
    trunk_roles = [
        role
        for role in roles
        if POSE_CONTROL_TO_ANATOMICAL_ANGLE.get(role) == "trunk_angle_deg"
    ]
    if trunk_roles and trunk_roles != ["lower_spine"]:
        raise RuntimeError(
            "Trunk target must have exactly one CARLA control bone; got "
            + ", ".join(trunk_roles)
        )
    forbidden = {"pelvis", "upper_spine", "middle_spine", "chest"}.intersection(roles)
    if forbidden:
        raise RuntimeError(
            "Cumulative trunk rotation is forbidden on: " + ", ".join(sorted(forbidden))
        )
    if "pelvis" in POSE_CONTROL_TO_ANATOMICAL_ANGLE:
        raise RuntimeError("Pelvis literature reference was incorrectly mapped to a control")


def _pose_errors(
    roles: Sequence[str], measured: AnatomicalAngles
) -> Dict[str, float]:
    errors: Dict[str, float] = {}
    for role in roles:
        field_name = POSE_CONTROL_TO_ANATOMICAL_ANGLE[role]
        target = LITERATURE_TARGETS[field_name]
        if target is not None:
            errors[role] = float(target) - _angle_value(measured, field_name)
    return errors


def _wrapped_angle_delta(final_value: float, baseline_value: float) -> float:
    return (final_value - baseline_value + 180.0) % 360.0 - 180.0


def print_spine_control_audit(
    baseline_records: Sequence[FullBoneRecord],
    controlled_transforms: Mapping[str, Any],
) -> None:
    mapping = map_driver_bones(baseline_records)
    print("\nSPINE CONTROL AUDIT")
    print("Trunk anatomical target is controlled from final landmarks, not Euler sums.")
    for role, label in (
        ("pelvis", "pelvis"),
        ("lower_spine", "lower spine"),
        ("upper_spine", "upper spine / chest"),
    ):
        baseline = mapping[role].relative_transform.rotation
        final_transform = controlled_transforms.get(role)
        final = baseline if final_transform is None else final_transform.rotation
        deltas = (
            _wrapped_angle_delta(float(final.pitch), float(baseline.pitch)),
            _wrapped_angle_delta(float(final.yaw), float(baseline.yaw)),
            _wrapped_angle_delta(float(final.roll), float(baseline.roll)),
        )
        print(
            f"  {label}: bone={mapping[role].name} "
            f"trunk-controlled={'YES' if role == 'lower_spine' and role in controlled_transforms else 'NO'} "
            f"relative Euler delta=({deltas[0]:+.3f}, {deltas[1]:+.3f}, "
            f"{deltas[2]:+.3f}) deg"
        )
    print("  middle spine: no distinct bone in the CARLA unified walker hierarchy")
    print("  chest: no distinct bone beyond crl_spine01__C in this hierarchy")
    print("  pelvis rearward-change reference 55.4 deg applied as Euler control: NO")
    print("  neck is optimized separately from trunk; head Euler control: NONE")


def print_trunk_posture_validation(
    before: Optional[AnatomicalAngles], after: AnatomicalAngles
) -> None:
    measured_trunk = _angle_value(after, "trunk_angle_deg")
    measured_neck = _angle_value(after, "neck_segment_angle_deg")
    residual = measured_trunk - TRUNK_TARGET_DEG
    neck_residual = measured_neck - NECK_TARGET_DEG
    passed = abs(residual) <= TRUNK_VALIDATION_TOLERANCE_DEG
    print("\nBEFORE (pre-optimization get_bones() landmark geometry)")
    if before is None:
        print("Measured trunk inclination: N/A")
        print("Measured neck inclination: N/A")
    else:
        print(
            f"Measured trunk inclination: "
            f"{_angle_value(before, 'trunk_angle_deg'):.3f} deg from vertical"
        )
        print(
            f"Measured neck inclination: "
            f"{_angle_value(before, 'neck_segment_angle_deg'):.3f} deg from vertical"
        )
    print("\nAFTER (final get_bones() landmark geometry)")
    print(f"Measured trunk inclination: {measured_trunk:.3f} deg from vertical")
    print(f"Measured neck inclination: {measured_neck:.3f} deg from vertical")
    print(f"Literature target neck: {NECK_TARGET_DEG:.1f} deg from vertical")
    print(f"Neck residual (measured - target): {neck_residual:+.3f} deg")
    print(f"Literature target trunk: {TRUNK_TARGET_DEG:.1f} deg from vertical")
    print(f"Residual (measured - target): {residual:+.3f} deg")
    print("\nTRUNK POSTURE VALIDATION")
    print(f"Literature target:\n{TRUNK_TARGET_DEG:.1f} deg from vertical")
    print(f"\nMeasured final trunk inclination:\n{measured_trunk:.3f} deg from vertical")
    print(f"\nResidual:\n{residual:+.3f} deg")
    print(
        "\nTarget band: "
        f"{TRUNK_TARGET_DEG - TRUNK_VALIDATION_TOLERANCE_DEG:.1f} to "
        f"{TRUNK_TARGET_DEG + TRUNK_VALIDATION_TOLERANCE_DEG:.1f} deg"
    )
    print(f"Validation: {'PASS' if passed else 'FAIL'}")


def solve_literature_pose(
    walker: Any,
    vehicle: Any,
    world: Any,
    stage: str,
    axis_map: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
) -> PoseSolveResult:
    roles = _stage_control_roles(stage)
    _validate_trunk_control_separation(roles)
    if not roles:
        records = read_full_bones(walker, world, args)
        angles = measure_anatomical_angles(vehicle, records)
        return PoseSolveResult(
            stage, True, 0, angles, {}, 0.0, baseline_angles=angles
        )
    baseline = read_full_bones(walker, world, args)
    mapping = map_driver_bones(baseline)
    transforms = {
        role: _copy_transform(mapping[role].relative_transform) for role in roles
    }
    measured = measure_anatomical_angles(vehicle, baseline)
    baseline_angles = measured
    max_error = float("inf")
    effective_tolerance = min(
        args.pose_angle_tolerance, TRUNK_VALIDATION_TOLERANCE_DEG
    )
    for iteration in range(1, args.pose_iterations + 1):
        controls = [(mapping[role].name, transforms[role]) for role in roles]
        _apply_parent_relative_controls(walker, controls)
        advance_world(world, args.pose_ticks, args.timeout)
        current_records = read_full_bones(walker, world, args)
        measured = measure_anatomical_angles(vehicle, current_records)
        errors = _pose_errors(roles, measured)
        max_error = max((abs(value) for value in errors.values()), default=0.0)
        print(
            f"Pose solver {stage}: iteration={iteration} "
            f"max_landmark_error={max_error:.3f} deg"
        )
        if max_error <= effective_tolerance:
            return PoseSolveResult(
                stage,
                True,
                iteration,
                measured,
                transforms,
                max_error,
                baseline_angles=baseline_angles,
            )
        for role, error in errors.items():
            item = axis_map[role]
            axis = str(item["selected_axis"])
            gain = float(item["anatomical_deg_per_carla_deg"])
            delta = max(-args.pose_max_step, min(args.pose_max_step, error / gain))
            transforms[role] = _relative_transform_with_axis_delta(
                transforms[role], axis, delta
            )
    controls = [(mapping[role].name, transforms[role]) for role in roles]
    _apply_parent_relative_controls(walker, controls)
    advance_world(world, args.pose_ticks, args.timeout)
    measured = measure_anatomical_angles(
        vehicle, read_full_bones(walker, world, args)
    )
    final_errors = _pose_errors(roles, measured)
    max_error = max((abs(value) for value in final_errors.values()), default=0.0)
    return PoseSolveResult(
        stage,
        max_error <= effective_tolerance,
        args.pose_iterations,
        measured,
        transforms,
        max_error,
        baseline_angles=baseline_angles,
    )


def align_pelvis_to_seat_target(
    state_root: WalkerRootLocal,
    seat_target: Point3,
    vehicle: Any,
    walker: Any,
    world: Any,
    args: argparse.Namespace,
) -> Tuple[SkeletonSnapshot, Point3]:
    before = calculate_skeleton_snapshot(vehicle, walker, world, args)
    pelvis = before.vehicle_local["pelvis"]
    delta = seat_target.minus(pelvis)
    state_root.x += delta.x
    state_root.y += delta.y
    state_root.z += delta.z
    after = apply_root_and_measure(world, vehicle, walker, state_root, args)
    residual = seat_target.minus(after.vehicle_local["pelvis"])
    print("Pelvis re-alignment (seat target is a geometry point, not H-point/SgRP):")
    print(f"  root translation dx={delta.x:+.6f} dy={delta.y:+.6f} dz={delta.z:+.6f} m")
    print(
        f"  residual dx={residual.x:+.6f} dy={residual.y:+.6f} "
        f"dz={residual.z:+.6f} m"
    )
    return after, residual


def _transform_to_pose_json(transform: Any) -> Dict[str, Any]:
    return {
        "location_m": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation_deg": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def save_validated_literature_pose(
    solve_result: PoseSolveResult,
    mapping: Mapping[str, FullBoneRecord],
    axis_map_path: Path,
    seat_target: Point3,
    pelvis_residual: Point3,
    output_path: Path,
) -> None:
    # Re-read the artifact here as a second gate; callers cannot obtain a
    # validated-scope JSON merely by passing an arbitrary path string.
    load_verified_axis_map(axis_map_path)
    if not solve_result.converged:
        raise RuntimeError("Pose solver did not converge; validated JSON will not be written")
    if solve_result.stage not in ("seated", "seated-driver"):
        raise RuntimeError("Validated driver pose requires torso plus lower-body stage")
    if "pelvis" in solve_result.controlled_relative_transforms:
        raise RuntimeError("Pelvis 55.4-degree reference must not become an Euler control")
    trunk_residual = abs(
        _angle_value(solve_result.measured_angles, "trunk_angle_deg")
        - TRUNK_TARGET_DEG
    )
    if trunk_residual > TRUNK_VALIDATION_TOLERANCE_DEG:
        raise RuntimeError(
            "Final pelvis-to-shoulder trunk residual exceeds +/-3 degrees"
        )
    neck_residual = abs(
        _angle_value(solve_result.measured_angles, "neck_segment_angle_deg")
        - NECK_TARGET_DEG
    )
    if neck_residual > NECK_VALIDATION_TOLERANCE_DEG:
        raise RuntimeError("Final neck-to-head landmark residual exceeds +/-3 degrees")
    if pelvis_residual.distance_to(Point3(0.0, 0.0, 0.0)) > 0.005:
        raise RuntimeError("Pelvis-to-seat-target residual exceeds 5 mm")
    bones = []
    for role, transform in solve_result.controlled_relative_transforms.items():
        bones.append(
            {
                "name": mapping[role].name,
                "anatomical_role": role,
                "relative": _transform_to_pose_json(transform),
            }
        )
    references = [
        {
            "source": "Kyung (2008), Virginia Tech dissertation, Chapter 7 Table 31",
            "url": "https://vtechworks.lib.vt.edu/bitstreams/ed9034a2-405c-44bb-b179-d0bcf7bcf5af/download",
            "variables": "bilateral hip, knee, ankle; shoulder/elbow reported but deferred",
        },
        {
            "source": "Reed et al. (2000), SAE 2000-01-2180",
            "url": "https://mreed.umtri.umich.edu/mreed/pubs/Reed_2000-01-2180.pdf",
            "variables": "torso and knee multivariate posture model",
        },
        {
            "source": "Reed et al. (2002), Human Factors 44(4), 557-568",
            "url": "https://mreed.umtri.umich.edu/mreed/pubs/Reed_2002.pdf",
            "variables": "package-dependent posture prediction and eye/hip priorities",
        },
        {
            "source": "Reed and Schneider (1996), SAE 960478",
            "url": "https://mreed.umtri.umich.edu/mreed/pubs/Reed_19960478.pdf",
            "variables": "pelvis and spine posture limitation",
        },
    ]
    payload = {
        "pose_name": "literature_based_driver_seated_pose",
        "walker": WALKER_BLUEPRINT,
        "vehicle_context": VEHICLE_BLUEPRINT,
        "anthropometric_population": "not Korean-P50 validated",
        "pose_basis": "automotive driver posture literature plus live CARLA axis mapping",
        "literature_pose_validated": True,
        "korean_female_anthropometry_validated": False,
        "metadata": {
            "source": "driver_seated_pose_references.md",
            "literature_pose_validated": True,
            "carla_bone_axis_validated": True,
            "seat_reference_calibrated": True,
            "anthropometrically_valid": False,
            "validation_scope": (
                "literature parameters + CARLA bone mapping + geometry seat target; "
                "not Korean female anthropometric validation"
            ),
            "axis_map": str(axis_map_path),
        },
        "references": references,
        "joint_targets": {
            field_name: LITERATURE_TARGETS[field_name]
            for field_name in POSE_CONTROL_TO_ANATOMICAL_ANGLE.values()
        },
        "reported_not_directly_applied": {
            "pelvis_rearward_change_from_standing_deg": LITERATURE_TARGETS[
                "pelvis_rearward_change_from_standing_deg"
            ],
            "head_angle_deg": None,
            "reason": (
                "The pelvis result is a landmark change, not a transferable CARLA "
                "Euler rotation; no compatible head-angle target was verified."
            ),
        },
        "measured_joint_angles": solve_result.measured_angles.as_dict(),
        "pose_stage": solve_result.stage,
        "solver_max_error_deg": solve_result.max_error_deg,
        "trunk_posture_validation": {
            "measurement": (
                "pelvis-to-bilateral-shoulder-midpoint vector from Prius vertical"
            ),
            "target_deg": TRUNK_TARGET_DEG,
            "measured_deg": _angle_value(
                solve_result.measured_angles, "trunk_angle_deg"
            ),
            "residual_measured_minus_target_deg": (
                _angle_value(solve_result.measured_angles, "trunk_angle_deg")
                - TRUNK_TARGET_DEG
            ),
            "tolerance_deg": TRUNK_VALIDATION_TOLERANCE_DEG,
            "passed": True,
        },
        "seat_target_vehicle_local": seat_target.as_dict(),
        "pelvis_alignment_residual_m": pelvis_residual.as_dict(),
        "arms_status": "deferred; Prius steering-wheel grip target not measured",
        "head_angle_status": "N/A; compatible literature head-angle target not established",
        "bones": bones,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Validated-scope literature pose saved: {output_path}")


def read_bones(walker: Any, world: Any, args: argparse.Namespace) -> List[BoneRecord]:
    last_error: Optional[BaseException] = None
    for _ in range(args.bone_retries):
        try:
            output = walker.get_bones()
            records = [
                BoneRecord(str(item.name), Point3.from_carla(item.world.location))
                for item in output.bone_transforms
            ]
            if records:
                return records
        except (AttributeError, RuntimeError, ValueError) as exc:
            last_error = exc
        advance_world(world, 1, args.timeout)
    if last_error is not None:
        raise RuntimeError(f"get_bones() failed: {last_error}")
    raise RuntimeError("get_bones() repeatedly returned no bones")


def calculate_skeleton_snapshot(
    vehicle: Any, walker: Any, world: Any, args: argparse.Namespace
) -> SkeletonSnapshot:
    records = read_bones(walker, world, args)
    selected = find_skeleton_bones(records)
    missing = [role for role in EXACT_BONES if role not in selected]
    if missing:
        names = ", ".join(record.name for record in records)
        raise RuntimeError(
            "Required skeleton bones not found: " + ", ".join(missing) + "; available=" + names
        )
    left_eye = selected["left_eye"].world
    right_eye = selected["right_eye"].world
    eye_mid = left_eye.midpoint(right_eye)
    world_points = {
        "pelvis": selected["pelvis"].world,
        "head": selected["head"].world,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "eye_mid": eye_mid,
    }
    vehicle_transform = vehicle.get_transform()
    local_points = {
        role: Point3.from_carla(
            world_to_vehicle_local(vehicle_transform, point.to_location())
        )
        for role, point in world_points.items()
    }
    return SkeletonSnapshot(
        world=world_points,
        vehicle_local=local_points,
        bone_names={role: selected[role].name for role in EXACT_BONES},
    )


def apply_root_and_measure(
    world: Any,
    vehicle: Any,
    walker: Any,
    root: WalkerRootLocal,
    args: argparse.Namespace,
) -> SkeletonSnapshot:
    keep_walker_at_vehicle_local_root(vehicle, walker, root)
    advance_world(world, args.pose_ticks, args.timeout)
    return calculate_skeleton_snapshot(vehicle, walker, world, args)


def keep_walker_at_vehicle_local_root(
    vehicle: Any, walker: Any, root: WalkerRootLocal
) -> None:
    """Reapply the saved Prius-local root using the current Prius transform."""

    world_transform = vehicle_local_transform_to_world(
        vehicle.get_transform(), root.to_transform()
    )
    walker.set_transform(world_transform)


def _look_at_rotation(origin: Point3, target: Point3) -> Any:
    delta = target.minus(origin)
    horizontal = math.hypot(delta.x, delta.y)
    return carla.Rotation(
        pitch=math.degrees(math.atan2(delta.z, horizontal)),
        yaw=math.degrees(math.atan2(delta.y, delta.x)),
        roll=0.0,
    )


def place_spectator(world: Any, vehicle: Any) -> None:
    vehicle_transform = vehicle.get_transform()
    origin = Point3.from_carla(
        vehicle_local_to_world(vehicle_transform, carla.Location(x=0.4, y=-4.2, z=2.0))
    )
    target = Point3.from_carla(
        vehicle_local_to_world(vehicle_transform, carla.Location(x=0.4, y=-0.3, z=0.9))
    )
    world.get_spectator().set_transform(
        carla.Transform(origin.to_location(), _look_at_rotation(origin, target))
    )


def draw_debug_points(world: Any, snapshot: SkeletonSnapshot, args: argparse.Namespace) -> None:
    definitions = (
        ("eye_mid", "EYE MID", carla.Color(255, 0, 0), args.debug_point_size * 1.2),
        ("left_eye", "LEFT EYE", carla.Color(255, 0, 255), args.debug_point_size),
        ("right_eye", "RIGHT EYE", carla.Color(255, 0, 255), args.debug_point_size),
        ("head", "HEAD", carla.Color(255, 255, 0), args.debug_point_size),
        ("pelvis", "PELVIS", carla.Color(0, 0, 255), args.debug_point_size),
    )
    for role, label, color, size in definitions:
        point = snapshot.world[role]
        world.debug.draw_point(
            point.to_location(),
            size=size,
            color=color,
            life_time=args.debug_life_time,
            persistent_lines=False,
        )
        text_location = carla.Location(x=point.x, y=point.y, z=point.z + 0.07)
        world.debug.draw_string(
            text_location,
            label,
            draw_shadow=True,
            color=color,
            life_time=args.debug_life_time,
            persistent_lines=False,
        )


def spawn_eye_camera(
    world: Any,
    vehicle: Any,
    eye_local: Point3,
    args: argparse.Namespace,
    frame_buffer: CameraFrameBuffer,
) -> Any:
    if np is None:
        raise RuntimeError("--preview-eye-camera requires numpy")
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(args.preview_width))
    blueprint.set_attribute("image_size_y", str(args.preview_height))
    blueprint.set_attribute("fov", str(args.fov))
    camera_transform = carla.Transform(
        eye_local.to_location(),
        carla.Rotation(
            pitch=args.camera_pitch,
            yaw=args.camera_yaw,
            roll=args.camera_roll,
        ),
    )
    camera = world.spawn_actor(
        blueprint,
        camera_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )
    camera.listen(frame_buffer.callback)
    return camera


def update_eye_camera(camera: Any, eye_local: Point3, args: argparse.Namespace) -> None:
    # In CARLA 0.9.16 an attached actor's set_transform is applied as a parent-
    # relative transform by the server. Camera position and orientation remain
    # separate variables here.
    camera.set_transform(
        carla.Transform(
            eye_local.to_location(),
            carla.Rotation(
                pitch=args.camera_pitch,
                yaw=args.camera_yaw,
                roll=args.camera_roll,
            ),
        )
    )


def _print_point(label: str, point: Point3) -> None:
    print(f"{label}:")
    print(f"x = {point.x:.6f} m ({point.x * 1000.0:.1f} mm)")
    print(f"y = {point.y:.6f} m ({point.y * 1000.0:.1f} mm)")
    print(f"z = {point.z:.6f} m ({point.z * 1000.0:.1f} mm)")
    print()


def print_calibration_report(state: RuntimeState) -> None:
    print()
    print("=" * 60)
    print("PRIUS DRIVER MANNEQUIN CALIBRATION")
    print("=" * 60)
    print()
    print(f"Vehicle:\n{VEHICLE_BLUEPRINT}\n")
    print(f"Walker:\n{WALKER_BLUEPRINT}\n")
    print("POSE STATUS:")
    if state.pose_validated:
        print("LITERATURE-SCOPE POSE VALIDATED (not Korean anthropometric validation)")
    else:
        print("NOT ANTHROPOMETRICALLY VALIDATED")
    print("Korean female P50 validated: NO")
    print(f"Pose source: {state.pose_description}\n")
    print("Walker root in Prius local coordinates:")
    print(f"x = {state.root.x:.6f} m")
    print(f"y = {state.root.y:.6f} m")
    print(f"z = {state.root.z:.6f} m")
    print(f"yaw = {state.root.yaw:.3f} deg")
    print()
    print("-" * 60)
    print("Skeleton points - Prius local")
    print("-" * 60)
    print()
    for role, label in (
        ("pelvis", "Pelvis"),
        ("head", "Head"),
        ("left_eye", "Left Eye"),
        ("right_eye", "Right Eye"),
    ):
        _print_point(label, state.snapshot.vehicle_local[role])
    print("EYE MIDPOINT - PRIUS LOCAL")
    _print_point("Eye Midpoint", state.snapshot.vehicle_local["eye_mid"])
    print("-" * 60)
    print("Skeleton points - World")
    print("-" * 60)
    print()
    for role, label in (
        ("pelvis", "Pelvis World"),
        ("head", "Head World"),
        ("left_eye", "Left Eye World"),
        ("right_eye", "Right Eye World"),
        ("eye_mid", "Eye Midpoint World"),
    ):
        _print_point(label, state.snapshot.world[role])

    print("MANNEQUIN EYE POINT")
    print("= measured from left/right skeleton eye bones")
    eye = state.snapshot.vehicle_local["eye_mid"]
    print(f"  ({eye.x:.6f}, {eye.y:.6f}, {eye.z:.6f}) m\n")
    print("DEMONSTRATION CAMERA")
    print("= optional HUD demonstration position; not automatically treated as truth")
    if state.compare_camera is None:
        print("  N/A (--compare-camera was not supplied)\n")
    else:
        camera = state.compare_camera
        difference = camera.minus(eye)
        distance = camera.distance_to(eye)
        _print_point("Current Camera", camera)
        _print_point("Mannequin Eye", eye)
        print("Difference camera - eye:")
        print(f"dx = {difference.x:.6f} m ({difference.x * 1000.0:.1f} mm)")
        print(f"dy = {difference.y:.6f} m ({difference.y * 1000.0:.1f} mm)")
        print(f"dz = {difference.z:.6f} m ({difference.z * 1000.0:.1f} mm)")
        print(f"3D offset = {distance:.6f} m ({distance * 1000.0:.1f} mm)\n")
    print("This is visual geometry calibration only: inspect pelvis/seat region,")
    print("torso/head cabin placement, windshield relation, and severe clipping.")
    print("Collision/physics is not used to declare a normal seated posture.")
    print("=" * 60)


def print_literature_seated_eye_report(
    snapshot: SkeletonSnapshot,
    angles: AnatomicalAngles,
    literature_pose_validated: bool,
) -> None:
    local = snapshot.vehicle_local
    standing = Point3(*NEUTRAL_STANDING_EYE_PRIUS_LOCAL)
    demo = Point3(*DEMONSTRATION_CAMERA_PRIUS_LOCAL)
    eye = local["eye_mid"]
    standing_shift = eye.minus(standing)
    demo_shift = demo.minus(eye)
    print()
    print("=" * 60)
    print("LITERATURE-BASED SEATED DRIVER EYE POINT")
    print("=" * 60)
    print(f"\nWalker:\n{WALKER_BLUEPRINT}\n")
    print("Pose:\nliterature-based seated-driver\n")
    print("Literature pose validated:")
    print("YES" if literature_pose_validated else "NO - CARLA BONE-AXIS VALIDATION REQUIRED")
    print("\nKorean female P50 validated:\nNO\n")
    for role, label in (
        ("pelvis", "Pelvis Prius-local"),
        ("head", "Head"),
        ("left_eye", "Left Eye"),
        ("right_eye", "Right Eye"),
        ("eye_mid", "Eye Midpoint"),
    ):
        _print_point(label, local[role])
    print("Measured landmark angles (not CARLA Euler rotations):")
    for name, value in angles.as_dict().items():
        print(f"  {name} = {'N/A' if value is None else f'{value:.3f} deg'}")
    print("\nNeutral standing reference:")
    print(f"  ({standing.x:.4f}, {standing.y:.4f}, {standing.z:.4f}) m")
    print("Literature seated-driver eye:")
    print(f"  ({eye.x:.4f}, {eye.y:.4f}, {eye.z:.4f}) m")
    print("Demonstration camera:")
    print(f"  ({demo.x:.4f}, {demo.y:.4f}, {demo.z:.4f}) m")
    print("Standing -> seated shift:")
    print(
        f"  dx={standing_shift.x:+.4f} dy={standing_shift.y:+.4f} "
        f"dz={standing_shift.z:+.4f} m"
    )
    print("Seated -> demonstration-camera shift (camera - eye):")
    print(
        f"  dx={demo_shift.x:+.4f} dy={demo_shift.y:+.4f} "
        f"dz={demo_shift.z:+.4f} m"
    )
    print(
        f"  3D offset={demo.distance_to(eye):.6f} m "
        f"({demo.distance_to(eye) * 1000.0:.1f} mm)"
    )
    print("\nEye position comes from skeleton geometry. Camera rotation and FOV are separate.")
    print("=" * 60)


def draw_driver_reference_points(
    world: Any,
    vehicle: Any,
    seated_snapshot: SkeletonSnapshot,
    seat_target: Optional[Point3],
    args: argparse.Namespace,
) -> None:
    vehicle_transform = vehicle.get_transform()
    neutral = Point3(*NEUTRAL_STANDING_EYE_PRIUS_LOCAL)
    demo = Point3(*DEMONSTRATION_CAMERA_PRIUS_LOCAL)
    seated = seated_snapshot.vehicle_local["eye_mid"]
    definitions = [
        (neutral, "NEUTRAL EYE", carla.Color(255, 255, 255)),
        (demo, "DEMO CAMERA", carla.Color(0, 255, 0)),
    ]
    if seat_target is not None:
        definitions.append((seat_target, "SEAT TARGET (NOT H-POINT)", carla.Color(0, 255, 255)))
    for point, label, color in definitions:
        world_point = vehicle_local_to_world(vehicle_transform, point.to_location())
        world.debug.draw_point(
            world_point,
            size=args.debug_point_size,
            color=color,
            life_time=args.debug_life_time,
            persistent_lines=False,
        )
        world.debug.draw_string(
            carla.Location(x=world_point.x, y=world_point.y, z=world_point.z + 0.08),
            label,
            draw_shadow=True,
            color=color,
            life_time=args.debug_life_time,
            persistent_lines=False,
        )
    for start, end, color in (
        (neutral, seated, carla.Color(255, 255, 255)),
        (seated, demo, carla.Color(0, 255, 0)),
    ):
        world.debug.draw_line(
            vehicle_local_to_world(vehicle_transform, start.to_location()),
            vehicle_local_to_world(vehicle_transform, end.to_location()),
            thickness=0.025,
            color=color,
            life_time=args.debug_life_time,
            persistent_lines=False,
        )


def draw_trunk_posture_validation(
    world: Any,
    vehicle: Any,
    posture: TrunkPostureGeometry,
    args: argparse.Namespace,
) -> None:
    """Draw vehicle vertical (cyan) and measured trunk vector (yellow)."""

    vehicle_transform = vehicle.get_transform()
    pelvis = posture.pelvis_local
    shoulder = posture.shoulder_mid_local
    trunk_length = pelvis.distance_to(shoulder)
    vertical_top = Point3(pelvis.x, pelvis.y, pelvis.z + trunk_length)
    pelvis_world = vehicle_local_to_world(vehicle_transform, pelvis.to_location())
    shoulder_world = vehicle_local_to_world(vehicle_transform, shoulder.to_location())
    vertical_world = vehicle_local_to_world(
        vehicle_transform, vertical_top.to_location()
    )
    cyan = carla.Color(0, 255, 255)
    yellow = carla.Color(255, 255, 0)
    world.debug.draw_line(
        pelvis_world,
        vertical_world,
        thickness=0.035,
        color=cyan,
        life_time=args.debug_life_time,
        persistent_lines=False,
    )
    world.debug.draw_string(
        vertical_world,
        "VEHICLE VERTICAL",
        draw_shadow=True,
        color=cyan,
        life_time=args.debug_life_time,
        persistent_lines=False,
    )
    world.debug.draw_line(
        pelvis_world,
        shoulder_world,
        thickness=0.055,
        color=yellow,
        life_time=args.debug_life_time,
        persistent_lines=False,
    )
    midpoint = pelvis.midpoint(shoulder)
    label_world = vehicle_local_to_world(
        vehicle_transform,
        Point3(midpoint.x, midpoint.y, midpoint.z + 0.08).to_location(),
    )
    world.debug.draw_string(
        label_world,
        f"TRUNK = {posture.trunk_inclination_deg:.1f} deg",
        draw_shadow=True,
        color=yellow,
        life_time=args.debug_life_time,
        persistent_lines=False,
    )
    target_world = vehicle_local_to_world(
        vehicle_transform,
        Point3(midpoint.x, midpoint.y, midpoint.z + 0.14).to_location(),
    )
    world.debug.draw_string(
        target_world,
        f"TARGET = {TRUNK_TARGET_DEG:.1f} deg",
        draw_shadow=True,
        color=yellow,
        life_time=args.debug_life_time,
        persistent_lines=False,
    )


def save_seated_driver_measurement(
    snapshot: SkeletonSnapshot,
    angles: AnatomicalAngles,
    literature_pose_validated: bool,
    pose_source: str,
    output_path: Path,
) -> None:
    local = snapshot.vehicle_local
    demo = Point3(*DEMONSTRATION_CAMERA_PRIUS_LOCAL)
    eye = local["eye_mid"]
    difference = demo.minus(eye)
    row: Dict[str, Any] = {
        "walker_id": WALKER_BLUEPRINT,
        "pose_source": pose_source,
        "literature_pose_validated": str(literature_pose_validated).lower(),
        "korean_female_validated": "false",
    }
    for role in ("pelvis", "head", "left_eye", "right_eye", "eye_mid"):
        point = local[role]
        for axis in ("x", "y", "z"):
            row[f"{role}_local_{axis}"] = _csv_value(getattr(point, axis))
    measured = angles.as_dict()
    row["trunk_angle_deg"] = _csv_value(measured["trunk_angle_deg"])
    for output_name, left_name, right_name in (
        ("hip_angle_deg", "left_hip_angle_deg", "right_hip_angle_deg"),
        ("knee_angle_deg", "left_knee_angle_deg", "right_knee_angle_deg"),
        ("ankle_angle_deg", "left_ankle_angle_deg", "right_ankle_angle_deg"),
    ):
        values = [measured[left_name], measured[right_name]]
        row[output_name] = _csv_value(
            None if any(value is None for value in values) else sum(values) / 2.0  # type: ignore[arg-type]
        )
    row["head_angle_deg"] = _csv_value(measured["head_angle_deg"])
    for axis in ("x", "y", "z"):
        row[f"demo_camera_{axis}"] = _csv_value(getattr(demo, axis))
        row[f"demo_eye_d{axis}"] = _csv_value(getattr(difference, axis))
    row["demo_eye_distance_m"] = _csv_value(demo.distance_to(eye))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SEATED_DRIVER_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(f"Seated driver eye CSV saved: {output_path}")


def _csv_value(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.9f}"


def save_calibration(
    state: RuntimeState,
    json_path: Path,
    csv_path: Path,
    seated_csv_path: Path = DEFAULT_SEATED_CSV_OUTPUT,
) -> None:
    points = state.snapshot.vehicle_local
    eye = points["eye_mid"]
    camera = state.compare_camera
    difference = None if camera is None else camera.minus(eye)
    distance = None if camera is None else camera.distance_to(eye)
    payload = {
        "vehicle": VEHICLE_BLUEPRINT,
        "walker": WALKER_BLUEPRINT,
        "pose_validated": state.pose_validated,
        "literature_pose_validated": state.literature_pose_validated,
        "korean_female_anthropometry_validated": False,
        "pose_stage": state.pose_stage,
        "pose_status": state.pose_description,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "walker_root_vehicle_local": state.root.as_dict(),
        "pelvis_vehicle_local": points["pelvis"].as_dict(),
        "head_vehicle_local": points["head"].as_dict(),
        "left_eye_vehicle_local": points["left_eye"].as_dict(),
        "right_eye_vehicle_local": points["right_eye"].as_dict(),
        "eye_mid_vehicle_local": eye.as_dict(),
        "demonstration_camera_vehicle_local": None if camera is None else camera.as_dict(),
        "camera_minus_eye": None if difference is None else difference.as_dict(),
        "camera_eye_distance_m": distance,
        "trunk_posture_validation": (
            None
            if state.trunk_posture is None
            else {
                "measurement": "pelvis-to-shoulder-midpoint from Prius vertical",
                "target_deg": TRUNK_TARGET_DEG,
                "measured_deg": state.trunk_posture.trunk_inclination_deg,
                "residual_measured_minus_target_deg": (
                    state.trunk_posture.trunk_inclination_deg - TRUNK_TARGET_DEG
                ),
                "tolerance_deg": TRUNK_VALIDATION_TOLERANCE_DEG,
            }
        ),
        "research_warning": (
            "Geometry calibration only. A validated seated posture and anthropometric "
            "reference are required before claiming a Korean female representative eye point."
        ),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    row: Dict[str, Any] = {
        "vehicle_id": VEHICLE_BLUEPRINT,
        "walker_id": WALKER_BLUEPRINT,
        "pose_validated": str(state.pose_validated).lower(),
        "walker_root_local_x": _csv_value(state.root.x),
        "walker_root_local_y": _csv_value(state.root.y),
        "walker_root_local_z": _csv_value(state.root.z),
        "walker_root_local_yaw": _csv_value(state.root.yaw),
    }
    for role in ("pelvis", "head", "left_eye", "right_eye", "eye_mid"):
        point = points[role]
        row[f"{role}_local_x"] = _csv_value(point.x)
        row[f"{role}_local_y"] = _csv_value(point.y)
        row[f"{role}_local_z"] = _csv_value(point.z)
    for axis in ("x", "y", "z"):
        row[f"compare_camera_{axis}"] = _csv_value(
            None if camera is None else getattr(camera, axis)
        )
        row[f"camera_eye_d{axis}"] = _csv_value(
            None if difference is None else getattr(difference, axis)
        )
    row["camera_eye_distance_m"] = _csv_value(distance)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    if state.anatomical_angles is not None and state.pose_stage != "neutral":
        save_seated_driver_measurement(
            state.snapshot,
            state.anatomical_angles,
            state.literature_pose_validated,
            state.pose_description,
            seated_csv_path,
        )
    state.saved_once = True
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def read_reference_measurement(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        print(f"WARNING: reference standing-measurement CSV not found: {path}")
        return None
    with path.open("r", encoding="utf-8", newline="") as stream:
        matches = [
            row for row in csv.DictReader(stream) if row.get("walker_id") == WALKER_BLUEPRINT
        ]
    if not matches:
        print(f"WARNING: {WALKER_BLUEPRINT} is absent from reference CSV: {path}")
        return None
    row = matches[0]
    print("Neutral-standing reference (not a seated eye-height calibration):")
    for field_name in (
        "standing_eye_height_m",
        "pelvis_to_eye_z_m",
        "pelvis_to_eye_3d_m",
        "seated_eye_height_m",
    ):
        print(f"  {field_name} = {row.get(field_name, 'N/A') or 'N/A'}")
    return row


def _adjust_root(root: WalkerRootLocal, key: str, coarse: bool) -> bool:
    position_step = 0.05 if coarse else 0.01
    yaw_step = 5.0 if coarse else 1.0
    changes = {
        "w": ("x", position_step),
        "s": ("x", -position_step),
        "a": ("y", -position_step),
        "d": ("y", position_step),
        "r": ("z", position_step),
        "f": ("z", -position_step),
        "q": ("yaw", -yaw_step),
        "e": ("yaw", yaw_step),
    }
    change = changes.get(key.casefold())
    if change is None:
        return False
    name, delta = change
    setattr(root, name, getattr(root, name) + delta)
    if name == "yaw":
        root.yaw = (root.yaw + 180.0) % 360.0 - 180.0
    return True


def _refresh_after_adjustment(
    state: RuntimeState,
    world: Any,
    vehicle: Any,
    walker: Any,
    camera: Optional[Any],
    args: argparse.Namespace,
) -> None:
    state.snapshot = apply_root_and_measure(world, vehicle, walker, state.root, args)
    if state.pose_stage != "neutral":
        records = read_full_bones(walker, world, args)
        state.anatomical_angles = measure_anatomical_angles(vehicle, records)
        state.trunk_posture = measure_trunk_posture_geometry(vehicle, records)
    if camera is not None:
        update_eye_camera(camera, state.snapshot.vehicle_local["eye_mid"], args)
    draw_debug_points(world, state.snapshot, args)
    if state.pose_stage != "neutral":
        draw_driver_reference_points(
            world, vehicle, state.snapshot, state.seat_target, args
        )
        if state.trunk_posture is not None:
            draw_trunk_posture_validation(
                world, vehicle, state.trunk_posture, args
            )
    print_calibration_report(state)


def run_text_calibration(
    state: RuntimeState,
    world: Any,
    vehicle: Any,
    walker: Any,
    camera: Optional[Any],
    args: argparse.Namespace,
) -> None:
    print("Text calibration controls: w/s x, a/d y, r/f z, q/e yaw")
    print("Uppercase uses coarse step. p or Enter saves; v prints; x exits.")
    while True:
        command = input("calibration> ").strip()
        if command.casefold() in ("x", "exit", "quit"):
            break
        if command == "" or command.casefold() in ("p", "save"):
            save_calibration(
                state, args.output_json, args.output_csv, args.seated_output_csv
            )
            continue
        if command.casefold() in ("v", "print"):
            print_calibration_report(state)
            continue
        if len(command) == 1 and _adjust_root(state.root, command, command.isupper()):
            _refresh_after_adjustment(state, world, vehicle, walker, camera, args)
        else:
            print("Unknown command")


def _pygame_lines(state: RuntimeState) -> List[str]:
    eye = state.snapshot.vehicle_local["eye_mid"]
    status = "EXTERNAL VALIDATED POSE" if state.pose_validated else "NOT ANTHROPOMETRICALLY VALIDATED"
    lines = [
        "PRIUS DRIVER MANNEQUIN CALIBRATION",
        f"POSE: {status}",
        f"Root local x={state.root.x:+.3f} y={state.root.y:+.3f} z={state.root.z:+.3f}",
        f"Root yaw={state.root.yaw:+.1f} deg",
        f"Eye local x={eye.x:+.4f} y={eye.y:+.4f} z={eye.z:+.4f} m",
        "W/S: +X/-X    A/D: -Y/+Y",
        "R/F: +Z/-Z    Q/E: yaw -/+",
        "Shift: coarse (0.05 m / 5 deg)",
        "Enter or P: save JSON + CSV",
        "Esc: quit",
        "Geometry calibration only; inspect clipping visually.",
    ]
    if state.compare_camera is not None:
        distance = state.compare_camera.distance_to(eye)
        lines.append(f"Demo-camera to eye offset: {distance * 1000.0:.1f} mm")
    if state.trunk_posture is not None:
        residual = state.trunk_posture.trunk_inclination_deg - TRUNK_TARGET_DEG
        lines.append(
            f"Trunk={state.trunk_posture.trunk_inclination_deg:.1f} deg "
            f"target={TRUNK_TARGET_DEG:.1f} residual={residual:+.1f}"
        )
    return lines


def run_pygame_calibration(
    state: RuntimeState,
    world: Any,
    vehicle: Any,
    walker: Any,
    camera: Optional[Any],
    frame_buffer: CameraFrameBuffer,
    args: argparse.Namespace,
) -> None:
    if pygame is None:
        raise RuntimeError("pygame is not installed")
    pygame.init()
    width = args.preview_width if args.preview_eye_camera else 900
    height = args.preview_height if args.preview_eye_camera else 520
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA Prius Driver Eye Calibration")
    font = pygame.font.Font(None, 27)
    clock = pygame.time.Clock()
    pygame.key.set_repeat(250, 55)
    running = True
    start = time.monotonic()
    last_debug = 0.0
    last_measurement = 0.0
    try:
        while running:
            now = time.monotonic()
            if args.run_seconds > 0.0 and now - start >= args.run_seconds:
                break
            changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_p):
                        save_calibration(
                            state,
                            args.output_json,
                            args.output_csv,
                            args.seated_output_csv,
                        )
                    else:
                        key_names = {
                            pygame.K_w: "w",
                            pygame.K_s: "s",
                            pygame.K_a: "a",
                            pygame.K_d: "d",
                            pygame.K_r: "r",
                            pygame.K_f: "f",
                            pygame.K_q: "q",
                            pygame.K_e: "e",
                        }
                        key_name = key_names.get(event.key)
                        coarse = bool(event.mod & pygame.KMOD_SHIFT)
                        if key_name is not None:
                            changed = _adjust_root(state.root, key_name, coarse) or changed
            if changed:
                _refresh_after_adjustment(state, world, vehicle, walker, camera, args)
                last_measurement = now
            else:
                # Keep the root state vehicle-local even if an external system
                # relocates the Prius during calibration.
                keep_walker_at_vehicle_local_root(vehicle, walker, state.root)
                advance_world(world, 1, args.timeout)
                if now - last_measurement >= args.measurement_refresh:
                    state.snapshot = calculate_skeleton_snapshot(vehicle, walker, world, args)
                    if camera is not None:
                        update_eye_camera(camera, state.snapshot.vehicle_local["eye_mid"], args)
                    last_measurement = now
            if now - last_debug >= max(0.1, args.debug_life_time * 0.5):
                draw_debug_points(world, state.snapshot, args)
                if state.pose_stage != "neutral":
                    draw_driver_reference_points(
                        world, vehicle, state.snapshot, state.seat_target, args
                    )
                    if state.trunk_posture is not None:
                        draw_trunk_posture_validation(
                            world, vehicle, state.trunk_posture, args
                        )
                last_debug = now

            screen.fill((18, 18, 22))
            if args.preview_eye_camera:
                camera_surface = frame_buffer.pygame_surface()
                if camera_surface is not None:
                    if camera_surface.get_size() != screen.get_size():
                        camera_surface = pygame.transform.smoothscale(camera_surface, screen.get_size())
                    screen.blit(camera_surface, (0, 0))
                    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 85))
                    screen.blit(overlay, (0, 0))
            y = 18
            for line in _pygame_lines(state):
                color = (255, 220, 90) if line.startswith("POSE:") else (240, 240, 240)
                surface = font.render(line, True, color)
                screen.blit(surface, (18, y))
                y += 29
            pygame.display.flip()
            clock.tick(args.ui_fps)
    finally:
        pygame.quit()


def run_non_interactive(
    state: RuntimeState,
    world: Any,
    vehicle: Any,
    walker: Any,
    camera: Optional[Any],
    args: argparse.Namespace,
) -> None:
    print_calibration_report(state)
    end = time.monotonic() + max(0.0, args.run_seconds)
    while time.monotonic() < end:
        keep_walker_at_vehicle_local_root(vehicle, walker, state.root)
        draw_debug_points(world, state.snapshot, args)
        if state.pose_stage != "neutral":
            draw_driver_reference_points(
                world, vehicle, state.snapshot, state.seat_target, args
            )
            if state.trunk_posture is not None:
                draw_trunk_posture_validation(
                    world, vehicle, state.trunk_posture, args
                )
        advance_world(world, 1, args.timeout)
    if camera is not None:
        update_eye_camera(camera, state.snapshot.vehicle_local["eye_mid"], args)


def cleanup_actors(camera: Optional[Any], walker: Optional[Any], vehicle: Optional[Any]) -> None:
    if camera is not None:
        try:
            if camera.is_listening:
                camera.stop()
        except (AttributeError, RuntimeError):
            pass
        try:
            if camera.is_alive:
                camera.destroy()
        except RuntimeError as exc:
            print(f"WARNING: camera cleanup failed: {exc}", file=sys.stderr)
    for label, actor in (("walker", walker), ("Prius", vehicle)):
        if actor is None:
            continue
        try:
            if actor.is_alive:
                actor.destroy()
        except RuntimeError as exc:
            print(f"WARNING: {label} cleanup failed: {exc}", file=sys.stderr)


def validate_api() -> None:
    if carla is None:
        raise RuntimeError(f"Could not import carla: {CARLA_IMPORT_ERROR}")
    required = {
        "carla.Actor.set_transform": hasattr(carla.Actor, "set_transform"),
        "carla.Actor.set_simulate_physics": hasattr(carla.Actor, "set_simulate_physics"),
        "carla.Actor.set_enable_gravity": hasattr(carla.Actor, "set_enable_gravity"),
        "carla.Actor.set_collisions": hasattr(carla.Actor, "set_collisions"),
        "carla.Walker.get_bones": hasattr(carla.Walker, "get_bones"),
        "carla.Walker.get_pose_from_animation": hasattr(
            carla.Walker, "get_pose_from_animation"
        ),
        "carla.Walker.set_bones": hasattr(carla.Walker, "set_bones"),
        "carla.Walker.blend_pose": hasattr(carla.Walker, "blend_pose"),
        "carla.Walker.show_pose": hasattr(carla.Walker, "show_pose"),
        "carla.WalkerBoneControlIn": hasattr(carla, "WalkerBoneControlIn"),
        "carla.Transform.inverse_transform": hasattr(carla.Transform, "inverse_transform"),
        "carla.AttachmentType.Rigid": hasattr(carla.AttachmentType, "Rigid"),
    }
    missing = [name for name, available in required.items() if not available]
    if missing:
        raise RuntimeError("CARLA Python API is missing: " + ", ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate walker.pedestrian.0022 eye bones in Toyota Prius coordinates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--max-spawn-attempts", type=int, default=40)
    parser.add_argument(
        "--root-x", type=float, default=0.140,
        help="Initial walker-root Prius-local X; a calibration starting point only.",
    )
    parser.add_argument(
        "--root-y", type=float, default=-0.350,
        help="Initial walker-root Prius-local Y; a calibration starting point only.",
    )
    parser.add_argument(
        "--root-z", type=float, default=0.640,
        help="Initial walker-root Prius-local Z; a calibration starting point only.",
    )
    parser.add_argument("--root-pitch", type=float, default=0.0)
    parser.add_argument("--root-yaw", type=float, default=0.0)
    parser.add_argument("--root-roll", type=float, default=0.0)
    parser.add_argument("--pose-ticks", type=int, default=3)
    parser.add_argument("--bone-retries", type=int, default=5)
    parser.add_argument("--seated-pose", type=Path)
    parser.add_argument(
        "--pose",
        choices=("neutral", "lower-body", "seated", "seated-driver"),
        default="neutral",
        help=(
            "Pose stage. Literature stages require a previously verified --axis-map; "
            "seated-driver currently defers steering-wheel-constrained arms."
        ),
    )
    parser.add_argument(
        "--bone-axis-test",
        action="store_true",
        help="Perturb parent-relative axes and interactively verify sagittal flexion.",
    )
    parser.add_argument("--axis-test-step", type=float, default=10.0)
    parser.add_argument("--axis-test-hold", type=float, default=0.4)
    parser.add_argument(
        "--bone-mapping-output", type=Path, default=DEFAULT_BONE_MAPPING_OUTPUT
    )
    parser.add_argument("--axis-map-output", type=Path, default=DEFAULT_AXIS_MAP_OUTPUT)
    parser.add_argument(
        "--axis-map",
        type=Path,
        default=DEFAULT_AXIS_MAP_OUTPUT,
        help="Human-confirmed axis map produced by --bone-axis-test.",
    )
    parser.add_argument(
        "--generate-literature-pose",
        action="store_true",
        help=(
            "Write a validated-scope pose only after axis verification, solver "
            "convergence, and pelvis/seat-target alignment."
        ),
    )
    parser.add_argument(
        "--validated-pose-output", type=Path, default=DEFAULT_VALIDATED_POSE_OUTPUT
    )
    parser.add_argument(
        "--seat-target",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Measured Prius-local pelvis geometry target. It is not asserted to be "
            "an H-point, R-point, or SgRP."
        ),
    )
    parser.add_argument("--pose-iterations", type=int, default=25)
    parser.add_argument("--pose-angle-tolerance", type=float, default=1.0)
    parser.add_argument("--pose-max-step", type=float, default=8.0)
    parser.add_argument(
        "--seated-output-csv", type=Path, default=DEFAULT_SEATED_CSV_OUTPUT
    )
    parser.add_argument("--compare-camera", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--preview-eye-camera", action="store_true")
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--preview-height", type=int, default=540)
    parser.add_argument("--debug-point-size", type=float, default=0.12)
    parser.add_argument("--debug-life-time", type=float, default=1.0)
    parser.add_argument("--measurement-refresh", type=float, default=0.25)
    parser.add_argument("--ui-fps", type=int, default=30)
    parser.add_argument("--text-ui", action="store_true")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Measure once without a UI; useful for automated runtime checks.",
    )
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0 or args.max_spawn_attempts < 1:
        raise ValueError("timeout and max spawn attempts must be positive")
    if args.pose_ticks < 1 or args.bone_retries < 1:
        raise ValueError("pose tick and bone retry counts must be positive")
    if not (1.0 <= args.fov <= 179.0):
        raise ValueError("--fov must be between 1 and 179 degrees")
    if args.preview_width < 64 or args.preview_height < 64:
        raise ValueError("preview dimensions must be at least 64 pixels")
    if args.debug_point_size <= 0.0 or args.debug_life_time <= 0.0:
        raise ValueError("debug point size/lifetime must be positive")
    if args.measurement_refresh <= 0.0 or args.ui_fps < 1 or args.run_seconds < 0.0:
        raise ValueError("refresh, FPS, and run duration arguments are invalid")
    if args.axis_test_step <= 0.0 or args.axis_test_hold < 0.0:
        raise ValueError("axis-test step must be positive and hold must be non-negative")
    if args.pose_iterations < 1:
        raise ValueError("--pose-iterations must be positive")
    if args.pose_angle_tolerance <= 0.0 or args.pose_max_step <= 0.0:
        raise ValueError("pose tolerance and maximum solver step must be positive")
    if args.generate_literature_pose:
        if args.pose not in ("seated", "seated-driver"):
            raise ValueError(
                "--generate-literature-pose requires --pose seated or seated-driver"
            )
        if args.seat_target is None:
            raise ValueError(
                "--generate-literature-pose requires a measured --seat-target X Y Z"
            )
    if args.bone_axis_test and args.seated_pose is not None:
        raise ValueError("--bone-axis-test cannot be combined with --seated-pose")
    if args.seated_pose is not None and args.pose != "neutral":
        raise ValueError(
            "--seated-pose already defines the pose; do not combine it with --pose"
        )
    if (
        args.seat_target is not None
        and args.pose == "neutral"
        and args.seated_pose is None
    ):
        raise ValueError("--seat-target requires a seated pose stage or --seated-pose")


def run(args: argparse.Namespace) -> int:
    validate_api()
    validate_args(args)
    print(IMPORTANT_NOTICE)
    print()
    print(LITERATURE_POSE_NOTICE)
    print()
    read_reference_measurement(args.reference_csv)

    pose_spec: Optional[SeatedPoseSpec] = None
    if args.seated_pose is not None:
        pose_spec = load_validated_seated_pose(args.seated_pose)
        print(f"Validated external seated pose loaded: {pose_spec.path}")
    else:
        print("POSE STATUS: NOT ANTHROPOMETRICALLY VALIDATED")
        if args.pose == "neutral" and not args.bone_axis_test:
            print("No seated pose selected; no arbitrary joint angles will be generated.")
    requested_stage = "seated-driver" if pose_spec is not None else args.pose

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    print(f"CARLA client: {client.get_client_version()}")
    print(f"CARLA server: {client.get_server_version()}")
    world = client.get_world()
    print(f"World: {world.get_map().name}")
    print(
        "World mode: "
        + ("synchronous (this tool supplies ticks)" if world.get_settings().synchronous_mode else "asynchronous")
    )

    vehicle = None
    walker = None
    camera = None
    frame_buffer = CameraFrameBuffer()
    try:
        vehicle, _ = spawn_prius(world, args)
        print(f"Spawned and fixed: {vehicle.type_id} actor_id={vehicle.id}")
        place_spectator(world, vehicle)
        walker = spawn_walker_near_prius(world, vehicle, args)
        print(f"Spawned near Prius: {walker.type_id} actor_id={walker.id}")

        root = WalkerRootLocal(
            args.root_x,
            args.root_y,
            args.root_z,
            args.root_pitch,
            args.root_yaw,
            args.root_roll,
        )
        print(
            "Initial root is a geometry-calibration starting point, not a final "
            "seat/H-point calibration."
        )
        initial_snapshot = apply_root_and_measure(world, vehicle, walker, root, args)
        neutral_full_records = read_full_bones(walker, world, args)
        dump_driver_bone_mapping(neutral_full_records, args.bone_mapping_output)

        if args.bone_axis_test:
            run_bone_axis_test(walker, vehicle, world, args)
            print("Axis-test mode complete; no seated pose or validated pose JSON was generated.")
            return 0

        pose_validated = False
        literature_pose_validated = False
        pose_description = "frozen neutral-standing CARLA animation snapshot"
        solve_result: Optional[PoseSolveResult] = None
        pelvis_residual: Optional[Point3] = None
        seat_target: Optional[Point3] = (
            None
            if args.seat_target is None
            else Point3(*[float(value) for value in args.seat_target])
        )

        if pose_spec is not None:
            apply_validated_seated_pose(walker, world, pose_spec, args)
            initial_snapshot = apply_root_and_measure(world, vehicle, walker, root, args)
            if seat_target is None:
                stored_target = pose_spec.metadata.get("seat_target_vehicle_local")
                if stored_target is not None:
                    seat_target = Point3(
                        *_float_triplet(stored_target, "seat_target_vehicle_local")
                    )
            source = str(pose_spec.metadata.get("source", pose_spec.path))
            pose_description = f"external validated-scope parent-relative pose: {source}"
        elif requested_stage != "neutral":
            print(f"Loading verified CARLA bone-axis map: {args.axis_map}")
            axis_map = load_verified_axis_map(args.axis_map)
            solve_result = solve_literature_pose(
                walker, vehicle, world, requested_stage, axis_map, args
            )
            initial_snapshot = calculate_skeleton_snapshot(vehicle, walker, world, args)
            pose_description = (
                f"literature landmark-angle solver stage={requested_stage}; "
                f"max error={solve_result.max_error_deg:.3f} deg"
            )
            if requested_stage == "seated-driver":
                print(
                    "ARMS STATUS: DEFERRED. No measured Prius steering-wheel grip "
                    "target is available, so cosmetic arm angles are not generated."
                )

        if requested_stage != "neutral" and seat_target is not None:
            initial_snapshot, pelvis_residual = align_pelvis_to_seat_target(
                root, seat_target, vehicle, walker, world, args
            )
        elif requested_stage != "neutral":
            print(
                "SEAT REFERENCE: N/A. Pose may be inspected, but its pelvis has not "
                "been aligned to a measured Prius seat target."
            )

        final_full_records = read_full_bones(walker, world, args)
        anatomical_angles = measure_anatomical_angles(vehicle, final_full_records)
        final_mapping = map_driver_bones(final_full_records)

        if solve_result is not None:
            residual_ok = (
                pelvis_residual is not None
                and pelvis_residual.distance_to(Point3(0.0, 0.0, 0.0)) <= 0.005
            )
            literature_pose_validated = bool(solve_result.converged and residual_ok)
            pose_validated = literature_pose_validated
            if args.generate_literature_pose:
                if seat_target is None or pelvis_residual is None:
                    raise RuntimeError(
                        "A measured seat target and residual are required before saving"
                    )
                save_validated_literature_pose(
                    solve_result,
                    final_mapping,
                    args.axis_map,
                    seat_target,
                    pelvis_residual,
                    args.validated_pose_output,
                )
        elif pose_spec is not None:
            metadata_literature_valid = (
                pose_spec.metadata.get("literature_pose_validated") is True
                and pose_spec.metadata.get("carla_bone_axis_validated") is True
                and pose_spec.metadata.get("seat_reference_calibrated") is True
            )
            residual_ok = (
                pelvis_residual is not None
                and pelvis_residual.distance_to(Point3(0.0, 0.0, 0.0)) <= 0.005
            )
            literature_pose_validated = bool(metadata_literature_valid and residual_ok)
            pose_validated = bool(pose_spec.validated and residual_ok)
            if not residual_ok:
                print(
                    "WARNING: external pose rotations were loaded, but no reproducible "
                    "pelvis/seat-target alignment was established in this run."
                )

        compare: Optional[Point3] = None
        if args.compare_camera is not None:
            compare = Point3(*[float(value) for value in args.compare_camera])
        elif requested_stage != "neutral":
            compare = Point3(*DEMONSTRATION_CAMERA_PRIUS_LOCAL)
        state = RuntimeState(
            root=root,
            pose_validated=pose_validated,
            pose_description=pose_description,
            snapshot=initial_snapshot,
            compare_camera=compare,
            anatomical_angles=anatomical_angles,
            literature_pose_validated=literature_pose_validated,
            pose_stage=requested_stage,
            seat_target=seat_target,
        )
        print("Skeleton bone mapping:")
        for role, name in state.snapshot.bone_names.items():
            print(f"  {role}: {name}")
        draw_debug_points(world, state.snapshot, args)
        if requested_stage != "neutral":
            draw_driver_reference_points(
                world, vehicle, state.snapshot, seat_target, args
            )
        print_calibration_report(state)
        if requested_stage in ("seated", "seated-driver"):
            print_literature_seated_eye_report(
                state.snapshot, anatomical_angles, literature_pose_validated
            )

        if args.preview_eye_camera:
            camera = spawn_eye_camera(
                world,
                vehicle,
                state.snapshot.vehicle_local["eye_mid"],
                args,
                frame_buffer,
            )
            print(
                "Eye RGB camera spawned with Rigid attachment. Position follows the "
                "MANNEQUIN EYE POINT; rotation/FOV remain separate preview settings."
            )

        if args.auto_save:
            save_calibration(
                state,
                args.output_json,
                args.output_csv,
                args.seated_output_csv,
            )

        if args.non_interactive:
            run_non_interactive(state, world, vehicle, walker, camera, args)
        elif args.text_ui:
            run_text_calibration(state, world, vehicle, walker, camera, args)
        else:
            try:
                run_pygame_calibration(
                    state, world, vehicle, walker, camera, frame_buffer, args
                )
            except (pygame.error if pygame is not None else RuntimeError) as exc:
                print(f"WARNING: Pygame UI unavailable ({exc}); using text UI.")
                run_text_calibration(state, world, vehicle, walker, camera, args)

        final_eye = state.snapshot.vehicle_local["eye_mid"]
        print()
        print("FINAL EYE MIDPOINT - PRIUS LOCAL")
        print(f"x={final_eye.x:.6f} m")
        print(f"y={final_eye.y:.6f} m")
        print(f"z={final_eye.z:.6f} m")
        if not state.saved_once:
            print("Calibration was not saved. Press Enter/P during interactive use or use --auto-save.")
        return 0
    finally:
        cleanup_actors(camera, walker, vehicle)
        print("Cleanup complete: created camera/walker/Prius actors destroyed.")
        print("World settings were not changed.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        print()
        print(LITERATURE_POSE_NOTICE)


if __name__ == "__main__":
    raise SystemExit(main())
