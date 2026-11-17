#!/usr/bin/env python3
"""Calibrate a Toyota Prius driver eye from the steering-wheel rotation centre.

The primary reference is never a seat, walker, vehicle bounding box, or an old
demonstration camera.  The program first inspects the spawned
``vehicle.toyota.prius`` for an explicitly named steering-wheel component,
bone, or socket.  If no semantic pivot is exposed, it enters manual calibration
with an *unresolved cursor*; (0, 0, 0) is merely the initial cursor origin and
is never saved or reported as a steering-wheel centre without confirmation.

CARLA/Prius local axes used here: +X forward, +Y right, +Z up.
"""

from __future__ import annotations

import argparse
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
except ImportError as exc:  # pragma: no cover - installation dependent
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
VEHICLE_LABEL = "Toyota Prius"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_FILE = SCRIPT_DIR / "prius_steering_wheel_center.json"
DEFAULT_API_AUDIT_FILE = SCRIPT_DIR / "prius_steering_wheel_api_audit.json"

# Measured at 1.055 m height: seatback -> monitor 0.910 m, seatback -> eye 0.200 m,
# wheel -> monitor 0.210 m. Hence eye -> monitor 0.710 m and eye -> wheel 0.500 m.
RIG_REARWARD_M = 0.500
RIG_UPWARD_M = 0.210
RIG_EYE_WHEEL_DISTANCE_M = math.hypot(RIG_REARWARD_M, RIG_UPWARD_M)
DEFAULT_LATERAL_M = 0.0
MONITOR_DISTANCE_M = 0.710
MONITOR_WIDTH_M = 0.6984
MONITOR_ASPECT_WIDTH = 16.0
MONITOR_ASPECT_HEIGHT = 9.0
FLAT_SCREEN_HORIZONTAL_FOV_DEG = math.degrees(
    2.0 * math.atan((MONITOR_WIDTH_M * 0.5) / MONITOR_DISTANCE_M)
)

# Thrustmaster T300RS axis/button layout, same as carla_wheel/drive_ego.py.
WHEEL_STEER_AXIS = 0
WHEEL_BRAKE_AXIS = 1
WHEEL_THROTTLE_AXIS = 2
WHEEL_REVERSE_BUTTON = 4
WHEEL_HANDBRAKE_BUTTON = 5
WHEEL_STEER_DEADZONE = 0.01
# Rotation range the T300RS reports over its full axis (its default, as in drive_ego.py).
WHEEL_RANGE_DEG = 1080.0

# Toyota Prius 3rd generation (ZVW30, 2010, 15in 195/65R15). Sources and the
# measured CARLA response are in prius_gen3_physics.md.
PRIUS_MASS_KG = 1380.0  # Toyota Canada 2010 brochure, curb weight
PRIUS_DRAG_COEFFICIENT = 0.25  # same brochure
PRIUS_STEERING_RATIO = 17.6  # 2010 European spec sheet, 15in wheels
PRIUS_LOCK_TO_LOCK_TURNS = 3.7  # same sheet; 3.7 * 180 / 17.6 = 37.8 deg road wheel
PRIUS_MAX_ROAD_WHEEL_DEG = PRIUS_LOCK_TO_LOCK_TURNS * 180.0 / PRIUS_STEERING_RATIO
PRIUS_TOP_SPEED_KMH = 180.0  # electronically limited
# The T300RS stops at +/-540 deg, short of the real +/-666 deg lock, so the wheel
# uses the ratio that puts the real road-wheel lock at the wheel's stop (14.27:1).
WHEEL_STEERING_RATIO = WHEEL_RANGE_DEG * 0.5 / PRIUS_MAX_ROAD_WHEEL_DEG
# Tuned against the measured CARLA response (prius_gen3_physics.md), not specs.
PRIUS_WHEEL_POWER_W = 117000.0
PRIUS_MAX_DRIVE_FORCE_N = 5000.0
PRIUS_BRAKE_TORQUE_NM = 1500.0
PRIUS_CREEP_SPEED_MPS = 7.0 / 3.6
PRIUS_CREEP_THROTTLE = 0.12


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float

    @classmethod
    def from_carla(cls, point: Any) -> "Point3":
        return cls(float(point.x), float(point.y), float(point.z))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Point3":
        return cls(float(value["x"]), float(value["y"]), float(value["z"]))

    def to_location(self) -> Any:
        return carla.Location(x=self.x, y=self.y, z=self.z)

    def plus(self, other: "Point3") -> "Point3":
        return Point3(self.x + other.x, self.y + other.y, self.z + other.z)

    def minus(self, other: "Point3") -> "Point3":
        return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance_to(self, other: "Point3") -> float:
        delta = self.minus(other)
        return math.sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)

    def as_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class CalibrationState:
    wheel_local: Point3
    source: str
    confirmed: bool
    step_m: float
    pitch_deg: float
    saved_once: bool = False
    camera_view: str = "monitor"


@dataclass(frozen=True)
class ApiCandidate:
    kind: str
    name: str
    world: Point3
    vehicle_local: Point3


class CameraFrameBuffer:
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


def calculate_flat_screen_horizontal_fov_deg(
    display_width_m: float, eye_to_display_m: float
) -> float:
    if display_width_m <= 0.0 or eye_to_display_m <= 0.0:
        raise ValueError("display width and viewing distance must be positive")
    return math.degrees(2.0 * math.atan((display_width_m * 0.5) / eye_to_display_m))


def calculate_driver_eye(
    wheel_local: Point3,
    rearward_m: float,
    lateral_m: float,
    upward_m: float,
) -> Point3:
    return Point3(
        wheel_local.x - rearward_m,
        wheel_local.y + lateral_m,
        wheel_local.z + upward_m,
    )


def calculate_virtual_monitor_center(
    eye_local: Point3,
    viewing_distance_m: float,
    lateral_offset_m: float = 0.0,
    vertical_offset_m: float = 0.0,
) -> Point3:
    """Place the flat virtual display forward of the eye in vehicle-local space."""

    return Point3(
        eye_local.x + viewing_distance_m,
        eye_local.y + lateral_offset_m,
        eye_local.z + vertical_offset_m,
    )


def calculate_monitor_height_m(screen_width_m: float) -> float:
    return screen_width_m * MONITOR_ASPECT_HEIGHT / MONITOR_ASPECT_WIDTH


def calculate_virtual_monitor_corners(
    center_local: Point3,
    screen_width_m: float,
) -> Tuple[Point3, Point3, Point3, Point3]:
    """Return TL, TR, BR, BL; plane horizontal is local Y, vertical is local Z."""

    half_width = screen_width_m * 0.5
    half_height = calculate_monitor_height_m(screen_width_m) * 0.5
    top_left = Point3(center_local.x, center_local.y - half_width, center_local.z + half_height)
    top_right = Point3(center_local.x, center_local.y + half_width, center_local.z + half_height)
    bottom_right = Point3(center_local.x, center_local.y + half_width, center_local.z - half_height)
    bottom_left = Point3(center_local.x, center_local.y - half_width, center_local.z - half_height)
    return top_left, top_right, bottom_right, bottom_left


def selected_camera_local(
    eye_local: Point3,
    state: CalibrationState,
    args: argparse.Namespace,
) -> Point3:
    if state.camera_view == "monitor":
        return calculate_virtual_monitor_center(
            eye_local,
            args.viewing_distance,
            args.monitor_lateral_offset,
            args.monitor_vertical_offset,
        )
    return eye_local


def toggle_camera_view(state: CalibrationState) -> str:
    state.camera_view = "monitor" if state.camera_view == "eye" else "eye"
    return state.camera_view


def vehicle_local_to_world(vehicle_transform: Any, local: Point3) -> Point3:
    location = local.to_location()
    vehicle_transform.transform(location)
    return Point3.from_carla(location)


def world_to_vehicle_local(vehicle_transform: Any, world: Point3) -> Point3:
    location = world.to_location()
    vehicle_transform.inverse_transform(location)
    return Point3.from_carla(location)


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


def _normalise_name(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


def _is_steering_wheel_name(name: str) -> bool:
    normalised = _normalise_name(name)
    return "steer" in normalised and "wheel" in normalised


def _transform_dict(transform: Any) -> Dict[str, Any]:
    return {
        "location": {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
        },
        "rotation": {
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
            "roll": float(transform.rotation.roll),
        },
    }


def inspect_steering_wheel_api(
    vehicle: Any,
    client_version: str,
    server_version: str,
    output_path: Path,
) -> Tuple[List[ApiCandidate], Dict[str, Any]]:
    """Inspect semantic component/bone/socket pivots without guessing geometry."""

    vehicle_transform = vehicle.get_transform()
    report: Dict[str, Any] = {
        "vehicle_blueprint": vehicle.type_id,
        "client_version": client_version,
        "server_version": server_version,
        "coordinate_frame": "vehicle_local: +X forward, +Y right, +Z up",
        "method_note": (
            "Only names containing both 'steer' and 'wheel' are candidates. "
            "Road-wheel metadata, vehicle bounds, seats, and walker points are excluded."
        ),
        "api_availability": {},
        "components": [],
        "bones": [],
        "sockets": [],
        "errors": [],
    }
    candidates: List[ApiCandidate] = []

    def inspect_indexed_group(
        kind: str,
        names_method: str,
        transforms_method: str,
    ) -> None:
        has_names = hasattr(vehicle, names_method)
        has_transforms = hasattr(vehicle, transforms_method)
        report["api_availability"][names_method] = has_names
        report["api_availability"][transforms_method] = has_transforms
        if not has_names or not has_transforms:
            return
        try:
            names = [str(name) for name in getattr(vehicle, names_method)()]
            transforms = list(getattr(vehicle, transforms_method)())
            if len(names) != len(transforms):
                raise RuntimeError(
                    f"{kind} name/transform count mismatch: {len(names)} != {len(transforms)}"
                )
            entries = []
            for name, transform in zip(names, transforms):
                world = Point3.from_carla(transform.location)
                local = world_to_vehicle_local(vehicle_transform, world)
                semantic_match = _is_steering_wheel_name(name)
                entries.append(
                    {
                        "name": name,
                        "semantic_steering_wheel_match": semantic_match,
                        "world_transform": _transform_dict(transform),
                        "vehicle_local_location": local.as_dict(),
                    }
                )
                if semantic_match:
                    candidates.append(ApiCandidate(kind, name, world, local))
            report[kind + "s"] = entries
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            report["errors"].append(f"{kind}: {type(exc).__name__}: {exc}")

    report["api_availability"]["get_component_names"] = hasattr(
        vehicle, "get_component_names"
    )
    report["api_availability"]["get_component_world_transform"] = hasattr(
        vehicle, "get_component_world_transform"
    )
    if report["api_availability"]["get_component_names"]:
        try:
            component_names = [str(name) for name in vehicle.get_component_names()]
            for name in component_names:
                entry: Dict[str, Any] = {
                    "name": name,
                    "semantic_steering_wheel_match": _is_steering_wheel_name(name),
                }
                if hasattr(vehicle, "get_component_world_transform"):
                    try:
                        transform = vehicle.get_component_world_transform(name)
                        world = Point3.from_carla(transform.location)
                        local = world_to_vehicle_local(vehicle_transform, world)
                        entry["world_transform"] = _transform_dict(transform)
                        entry["vehicle_local_location"] = local.as_dict()
                        if entry["semantic_steering_wheel_match"]:
                            candidates.append(ApiCandidate("component", name, world, local))
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        entry["transform_error"] = f"{type(exc).__name__}: {exc}"
                report["components"].append(entry)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            report["errors"].append(f"component names: {type(exc).__name__}: {exc}")

    inspect_indexed_group(
        "bone", "get_bone_names", "get_bone_world_transforms"
    )
    inspect_indexed_group(
        "socket", "get_socket_names", "get_socket_world_transforms"
    )
    report["steering_wheel_candidates"] = [
        {
            "kind": candidate.kind,
            "name": candidate.name,
            "world": candidate.world.as_dict(),
            "vehicle_local": candidate.vehicle_local.as_dict(),
        }
        for candidate in candidates
    ]
    report["automatic_result"] = (
        "one semantic candidate; pivot still requires visual confirmation"
        if len(candidates) == 1
        else "not uniquely extractable"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return candidates, report


def select_api_candidate(candidates: Sequence[ApiCandidate]) -> Optional[ApiCandidate]:
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    # A uniquely named semantic pivot is useful, but it is still shown for
    # visual confirmation because component origins are asset-author choices.
    return candidate


def load_calibration(path: Path) -> Optional[Tuple[Point3, str]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("vehicle_blueprint") != VEHICLE_BLUEPRINT:
        raise ValueError(f"Calibration file is not for {VEHICLE_BLUEPRINT}: {path}")
    if payload.get("coordinate_frame") != "vehicle_local":
        raise ValueError(f"Calibration file coordinate frame is not vehicle_local: {path}")
    if payload.get("confirmed") is not True:
        raise ValueError(f"Calibration file is not confirmed: {path}")
    centre = payload.get("steering_wheel_center")
    if not isinstance(centre, Mapping):
        raise ValueError(f"Calibration file has no steering_wheel_center: {path}")
    return Point3.from_mapping(centre), str(payload.get("source", "saved_manual"))


def save_calibration(
    path: Path,
    state: CalibrationState,
    eye_local: Point3,
    args: argparse.Namespace,
    client_version: str,
    server_version: str,
) -> None:
    offset = eye_local.minus(state.wheel_local)
    monitor_local = calculate_virtual_monitor_center(
        eye_local,
        args.viewing_distance,
        args.monitor_lateral_offset,
        args.monitor_vertical_offset,
    )
    payload = {
        "vehicle": VEHICLE_LABEL,
        "vehicle_blueprint": VEHICLE_BLUEPRINT,
        "carla_version": server_version,
        "carla_client_version": client_version,
        "coordinate_frame": "vehicle_local",
        "axis_convention": "+X forward, +Y right, +Z up",
        "confirmed": True,
        "source": state.source,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "steering_wheel_center_definition": (
            "geometric centre of steering-wheel rotation axis; not a rim edge"
        ),
        "steering_wheel_center": state.wheel_local.as_dict(),
        "physical_rig_eye_offset": {
            "dx_m": offset.x,
            "dy_m": offset.y,
            "dz_m": offset.z,
            "distance_m": state.wheel_local.distance_to(eye_local),
            "rearward_m": args.rearward,
            "lateral_m": args.lateral_offset,
            "upward_m": args.upward,
        },
        "calculated_driver_eye": eye_local.as_dict(),
        "virtual_monitor": {
            "center_vehicle_local": monitor_local.as_dict(),
            "viewing_distance_m": args.viewing_distance,
            "lateral_offset_m": args.monitor_lateral_offset,
            "vertical_offset_m": args.monitor_vertical_offset,
            "width_m": args.screen_width,
            "height_m": calculate_monitor_height_m(args.screen_width),
            "plane_axes": "horizontal=vehicle local Y, vertical=vehicle local Z, normal=-X",
        },
        "camera": {
            "position_mode": state.camera_view,
            "position_vehicle_local": selected_camera_local(
                eye_local, state, args
            ).as_dict(),
            "pitch_deg": state.pitch_deg,
            "yaw_deg": args.yaw,
            "roll_deg": args.roll,
            "horizontal_fov_deg": args.fov,
        },
        "flat_screen_fov_baseline": {
            "monitor_width_m": args.screen_width,
            "eye_to_monitor_m": args.viewing_distance,
            "horizontal_fov_deg": calculate_flat_screen_horizontal_fov_deg(
                args.screen_width, args.viewing_distance
            ),
            "curved_display_note": "1500R curvature is not included",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    state.confirmed = True
    state.saved_once = True
    print(f"Calibration saved: {path}")


def spawn_fixed_prius(world: Any, args: argparse.Namespace) -> Any:
    blueprint = world.get_blueprint_library().find(VEHICLE_BLUEPRINT)
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("Current map has no vehicle spawn points")
    start = args.spawn_index % len(spawn_points)
    candidates = spawn_points[start:] + spawn_points[:start]
    vehicle = None
    selected = None
    for transform in candidates[: args.max_spawn_attempts]:
        selected = _copy_transform(transform)
        vehicle = world.try_spawn_actor(blueprint, selected)
        if vehicle is not None:
            break
    if vehicle is None or selected is None:
        raise RuntimeError(f"Could not spawn {VEHICLE_BLUEPRINT}")
    try:
        vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
        )
        advance_world(world, 3, args.timeout)
        vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        vehicle.set_simulate_physics(False)
        vehicle.set_enable_gravity(False)
        vehicle.set_transform(selected)
        advance_world(world, 2, args.timeout)
        return vehicle
    except BaseException:
        if vehicle.is_alive:
            vehicle.destroy()
        raise


def _look_at_rotation(origin: Point3, target: Point3) -> Any:
    delta = target.minus(origin)
    horizontal = math.hypot(delta.x, delta.y)
    return carla.Rotation(
        pitch=math.degrees(math.atan2(delta.z, horizontal)),
        yaw=math.degrees(math.atan2(delta.y, delta.x)),
        roll=0.0,
    )


def place_spectator(world: Any, vehicle: Any, target_local: Point3) -> None:
    transform = vehicle.get_transform()
    origin = vehicle_local_to_world(transform, Point3(0.3, -3.0, 1.7))
    target = vehicle_local_to_world(transform, target_local)
    world.get_spectator().set_transform(
        carla.Transform(origin.to_location(), _look_at_rotation(origin, target))
    )


def print_measurement_report(
    state: CalibrationState,
    vehicle: Any,
    args: argparse.Namespace,
) -> Point3:
    eye = calculate_driver_eye(
        state.wheel_local,
        args.rearward,
        args.lateral_offset,
        args.upward,
    )
    wheel_world = vehicle_local_to_world(vehicle.get_transform(), state.wheel_local)
    eye_world = vehicle_local_to_world(vehicle.get_transform(), eye)
    monitor = calculate_virtual_monitor_center(
        eye,
        args.viewing_distance,
        args.monitor_lateral_offset,
        args.monitor_vertical_offset,
    )
    eye_monitor = monitor.minus(eye)
    eye_monitor_distance = eye.distance_to(monitor)
    monitor_height = calculate_monitor_height_m(args.screen_width)
    camera_local = selected_camera_local(eye, state, args)
    geometric_hfov = calculate_flat_screen_horizontal_fov_deg(
        args.screen_width, args.viewing_distance
    )
    offset = eye.minus(state.wheel_local)
    distance = state.wheel_local.distance_to(eye)
    print()
    print("=" * 60)
    print("PRIUS WHEEL-BASED DRIVER EYE CALIBRATION")
    print("=" * 60)
    print(f"Status: {'CONFIRMED' if state.confirmed else 'UNRESOLVED CURSOR / CONFIRM REQUIRED'}")
    print(f"Source: {state.source}\n")
    print("Prius steering wheel center - vehicle local")
    print(f"X = {state.wheel_local.x:.6f} m")
    print(f"Y = {state.wheel_local.y:.6f} m")
    print(f"Z = {state.wheel_local.z:.6f} m\n")
    print("Steering wheel center - world")
    print(f"X = {wheel_world.x:.6f} m")
    print(f"Y = {wheel_world.y:.6f} m")
    print(f"Z = {wheel_world.z:.6f} m\n")
    print("Measured physical rig offset")
    print(f"rearward = {args.rearward:.6f} m")
    print(f"lateral  = {args.lateral_offset:+.6f} m")
    print(f"upward   = {args.upward:.6f} m")
    print()
    print("Calculated driver eye - vehicle local")
    print(f"X = {eye.x:.6f} m")
    print(f"Y = {eye.y:.6f} m")
    print(f"Z = {eye.z:.6f} m\n")
    print("Calculated driver eye - world")
    print(f"X = {eye_world.x:.6f} m")
    print(f"Y = {eye_world.y:.6f} m")
    print(f"Z = {eye_world.z:.6f} m\n")
    print("Wheel -> eye verification (vehicle local, not world subtraction)")
    print(f"dx = {offset.x:+.6f} m (expected {-args.rearward:+.6f})")
    print(f"dy = {offset.y:+.6f} m (expected {args.lateral_offset:+.6f})")
    print(f"dz = {offset.z:+.6f} m (expected {args.upward:+.6f})")
    print(f"3D distance = {distance:.6f} m ({distance * 1000.0:.1f} mm)")
    print(
        f"Difference from rig {RIG_EYE_WHEEL_DISTANCE_M:.3f} m = "
        f"{(distance - RIG_EYE_WHEEL_DISTANCE_M) * 1000.0:+.2f} mm\n"
    )
    print("Horizontal FOV")
    print(f"Flat-screen geometric baseline = {geometric_hfov:.6f} deg")
    print(f"Camera configured FOV = {args.fov:.6f} deg")
    print(
        f"Basis: 2*atan(({args.screen_width:.4f}/2)/{args.viewing_distance:.4f})"
    )
    print("MSI MAG 32C6X 1500R curvature is not included in this baseline.")
    print("Camera position, pitch/yaw/roll, and FOV are independent variables.")
    print("=" * 60)
    print("Prius wheel-based driver-eye geometry")
    print(f"Wheel center local: ({state.wheel_local.x:+.6f}, {state.wheel_local.y:+.6f}, {state.wheel_local.z:+.6f}) m")
    print(f"Driver eye local:  ({eye.x:+.6f}, {eye.y:+.6f}, {eye.z:+.6f}) m")
    print(f"Virtual monitor center local: ({monitor.x:+.6f}, {monitor.y:+.6f}, {monitor.z:+.6f}) m")
    print("Eye -> monitor vector local:")
    print(f"dx = {eye_monitor.x:+.6f} m")
    print(f"dy = {eye_monitor.y:+.6f} m")
    print(f"dz = {eye_monitor.z:+.6f} m")
    print(f"Eye -> monitor distance = {eye_monitor_distance:.6f} m")
    print(f"Expected viewing distance = {args.viewing_distance:.6f} m")
    print(f"Monitor width = {args.screen_width:.6f} m")
    print(f"Monitor height = {monitor_height:.6f} m (16:9)")
    print(f"Flat geometric HFOV = {geometric_hfov:.6f} deg")
    print(f"Camera pitch = {state.pitch_deg:.6f} deg")
    vector_ok = (
        abs(eye_monitor.x - args.viewing_distance) <= 1e-9
        and abs(eye_monitor.y - args.monitor_lateral_offset) <= 1e-9
        and abs(eye_monitor.z - args.monitor_vertical_offset) <= 1e-9
    )
    distance_expected = math.sqrt(
        args.viewing_distance**2
        + args.monitor_lateral_offset**2
        + args.monitor_vertical_offset**2
    )
    distance_ok = abs(eye_monitor_distance - distance_expected) <= 1e-9
    print(f"Monitor vector validation = {'PASS' if vector_ok else 'FAIL'}")
    print(f"Monitor distance validation = {'PASS' if distance_ok else 'FAIL'}")
    print(f"Active camera view = {state.camera_view.upper()}")
    print(
        f"Active camera local = ({camera_local.x:+.6f}, "
        f"{camera_local.y:+.6f}, {camera_local.z:+.6f}) m"
    )
    print("M toggles camera position only; pitch/yaw/roll/FOV stay unchanged.")
    print("=" * 60)
    return eye


def draw_debug_geometry(
    world: Any,
    vehicle: Any,
    state: CalibrationState,
    args: argparse.Namespace,
) -> None:
    # CARLA debug primitives do not expose a non-emissive material.  The clean
    # preview path returns here before creating any primitive at all.
    if not args.debug_geometry:
        return
    transform = vehicle.get_transform()
    wheel = state.wheel_local
    eye = calculate_driver_eye(
        wheel, args.rearward, args.lateral_offset, args.upward
    )
    wheel_world = vehicle_local_to_world(transform, wheel)
    eye_world = vehicle_local_to_world(transform, eye)
    monitor = calculate_virtual_monitor_center(
        eye,
        args.viewing_distance,
        args.monitor_lateral_offset,
        args.monitor_vertical_offset,
    )
    monitor_world = vehicle_local_to_world(transform, monitor)
    lifetime = args.debug_life_time
    wheel_color = carla.Color(55, 85, 85)
    eye_color = carla.Color(85, 55, 85)
    world.debug.draw_point(
        wheel_world.to_location(),
        size=args.debug_point_size,
        color=wheel_color,
        life_time=lifetime,
        persistent_lines=False,
    )
    world.debug.draw_string(
        Point3(wheel_world.x, wheel_world.y, wheel_world.z + 0.05).to_location(),
        "STEERING ROTATION CENTER" if state.confirmed else "UNRESOLVED WHEEL CURSOR",
        draw_shadow=False,
        color=wheel_color,
        life_time=lifetime,
        persistent_lines=False,
    )
    world.debug.draw_point(
        eye_world.to_location(),
        size=args.debug_point_size,
        color=eye_color,
        life_time=lifetime,
        persistent_lines=False,
    )
    world.debug.draw_string(
        Point3(eye_world.x, eye_world.y, eye_world.z + 0.05).to_location(),
        "CALCULATED DRIVER EYE",
        draw_shadow=False,
        color=eye_color,
        life_time=lifetime,
        persistent_lines=False,
    )
    world.debug.draw_line(
        wheel_world.to_location(),
        eye_world.to_location(),
        thickness=args.guide_line_thickness,
        color=carla.Color(60, 60, 60),
        life_time=lifetime,
        persistent_lines=False,
    )
    axis_length = args.marker_axis_length
    for delta, color, label in (
        (Point3(axis_length, 0.0, 0.0), carla.Color(85, 35, 35), "+X"),
        (Point3(0.0, axis_length, 0.0), carla.Color(35, 85, 35), "+Y"),
        (Point3(0.0, 0.0, axis_length), carla.Color(35, 35, 85), "+Z"),
    ):
        endpoint = vehicle_local_to_world(transform, wheel.plus(delta))
        world.debug.draw_line(
            wheel_world.to_location(),
            endpoint.to_location(),
            thickness=args.guide_line_thickness,
            color=color,
            life_time=lifetime,
            persistent_lines=False,
        )
        world.debug.draw_string(
            endpoint.to_location(),
            label,
            draw_shadow=False,
            color=color,
            life_time=lifetime,
            persistent_lines=False,
        )
    pitch_rad = math.radians(state.pitch_deg)
    yaw_rad = math.radians(args.yaw)
    forward = Point3(
        args.forward_line_length * math.cos(pitch_rad) * math.cos(yaw_rad),
        args.forward_line_length * math.cos(pitch_rad) * math.sin(yaw_rad),
        args.forward_line_length * math.sin(pitch_rad),
    )
    if not args.hide_gaze_line:
        forward_end = vehicle_local_to_world(transform, eye.plus(forward))
        gaze_gray = args.gaze_line_gray
        gaze_color = carla.Color(gaze_gray, gaze_gray, gaze_gray)
        world.debug.draw_line(
            eye_world.to_location(),
            forward_end.to_location(),
            thickness=args.gaze_line_thickness,
            color=gaze_color,
            life_time=lifetime,
            persistent_lines=False,
        )
        if not args.hide_gaze_label:
            world.debug.draw_string(
                forward_end.to_location(),
                "GAZE DIRECTION",
                draw_shadow=False,
                color=gaze_color,
                life_time=lifetime,
                persistent_lines=False,
            )

    monitor_color = carla.Color(70, 70, 70)
    corners_local = calculate_virtual_monitor_corners(monitor, args.screen_width)
    corners_world = [
        vehicle_local_to_world(transform, corner) for corner in corners_local
    ]
    for start, end in zip(corners_world, corners_world[1:] + corners_world[:1]):
        world.debug.draw_line(
            start.to_location(),
            end.to_location(),
            thickness=args.monitor_line_thickness,
            color=monitor_color,
            life_time=lifetime,
            persistent_lines=False,
        )
    world.debug.draw_point(
        monitor_world.to_location(),
        size=args.monitor_center_point_size,
        color=monitor_color,
        life_time=lifetime,
        persistent_lines=False,
    )
    world.debug.draw_line(
        eye_world.to_location(),
        monitor_world.to_location(),
        thickness=args.monitor_line_thickness,
        color=monitor_color,
        life_time=lifetime,
        persistent_lines=False,
    )


def spawn_eye_camera(
    world: Any,
    vehicle: Any,
    eye_local: Point3,
    state: CalibrationState,
    args: argparse.Namespace,
    frame_buffer: CameraFrameBuffer,
) -> Any:
    if np is None:
        raise RuntimeError("--preview-eye-camera requires numpy")
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(args.preview_width))
    blueprint.set_attribute("image_size_y", str(args.preview_height))
    blueprint.set_attribute("fov", str(args.fov))
    camera = world.spawn_actor(
        blueprint,
        carla.Transform(
            eye_local.to_location(),
            carla.Rotation(
                pitch=state.pitch_deg,
                yaw=args.yaw,
                roll=args.roll,
            ),
        ),
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )
    camera.listen(frame_buffer.callback)
    return camera


def update_eye_camera(
    camera: Any,
    eye_local: Point3,
    state: CalibrationState,
    args: argparse.Namespace,
) -> None:
    camera.set_transform(
        carla.Transform(
            eye_local.to_location(),
            carla.Rotation(
                pitch=state.pitch_deg,
                yaw=args.yaw,
                roll=args.roll,
            ),
        )
    )


def adjust_state(state: CalibrationState, key: str) -> bool:
    delta = {
        "w": Point3(state.step_m, 0.0, 0.0),
        "s": Point3(-state.step_m, 0.0, 0.0),
        "a": Point3(0.0, -state.step_m, 0.0),
        "d": Point3(0.0, state.step_m, 0.0),
        "r": Point3(0.0, 0.0, state.step_m),
        "f": Point3(0.0, 0.0, -state.step_m),
    }.get(key.casefold())
    if delta is None:
        return False
    state.wheel_local = state.wheel_local.plus(delta)
    state.confirmed = False
    state.source = "manual_interactive"
    return True


def confirm_manual_cursor(state: CalibrationState) -> None:
    if state.source == "vehicle_origin_cursor_not_a_wheel_measurement":
        raise ValueError(
            "The untouched vehicle-origin cursor cannot be confirmed as a wheel centre. "
            "Move it, use --initial-wheel/--wheel-center, or use a semantic API pivot."
        )
    state.confirmed = True
    if not state.source.startswith("api_") and state.source != "user_cli_measured":
        state.source = "manual_interactive"


def _refresh(
    state: CalibrationState,
    world: Any,
    vehicle: Any,
    camera: Optional[Any],
    args: argparse.Namespace,
) -> Point3:
    eye = print_measurement_report(state, vehicle, args)
    draw_debug_geometry(world, vehicle, state, args)
    if camera is not None:
        update_eye_camera(camera, selected_camera_local(eye, state, args), state, args)
    return eye


def run_text_ui(
    state: CalibrationState,
    world: Any,
    vehicle: Any,
    camera: Optional[Any],
    args: argparse.Namespace,
    client_version: str,
    server_version: str,
) -> None:
    print("\nText controls: w/s X, a/d Y, r/f Z, 1=10mm, 2=5mm, 3=1mm")
    print("m toggles DRIVER EYE / MONITOR CENTER camera position")
    print("pitch+ / pitch- adjusts camera pitch only; p or Enter confirms+saves; x exits")
    while True:
        command = input("wheel-calibration> ").strip().casefold()
        if command in ("x", "exit", "quit"):
            return
        if command in ("1", "2", "3"):
            state.step_m = {"1": 0.010, "2": 0.005, "3": 0.001}[command]
            print(f"Position step = {state.step_m * 1000.0:.0f} mm")
            continue
        if command in ("pitch+", "pitch-"):
            state.pitch_deg += args.pitch_step * (1.0 if command == "pitch+" else -1.0)
            _refresh(state, world, vehicle, camera, args)
            continue
        if command in ("m", "view", "toggle"):
            view = toggle_camera_view(state)
            eye = _refresh(state, world, vehicle, camera, args)
            camera_local = selected_camera_local(eye, state, args)
            print(
                f"CAMERA VIEW = {view.upper()}: "
                f"x={camera_local.x:.6f}, y={camera_local.y:.6f}, "
                f"z={camera_local.z:.6f} m"
            )
            continue
        if command in ("", "p", "save", "confirm"):
            try:
                confirm_manual_cursor(state)
            except ValueError as exc:
                print(f"REFUSED: {exc}")
                continue
            eye = print_measurement_report(state, vehicle, args)
            save_calibration(
                args.calibration_file,
                state,
                eye,
                args,
                client_version,
                server_version,
            )
            continue
        if command in ("v", "print"):
            print_measurement_report(state, vehicle, args)
            continue
        if len(command) == 1 and adjust_state(state, command):
            _refresh(state, world, vehicle, camera, args)
            continue
        print("Unknown command")


def _pygame_lines(
    state: CalibrationState,
    eye: Point3,
    wheel_world: Point3,
    args: argparse.Namespace,
) -> List[str]:
    distance = state.wheel_local.distance_to(eye)
    monitor = calculate_virtual_monitor_center(
        eye,
        args.viewing_distance,
        args.monitor_lateral_offset,
        args.monitor_vertical_offset,
    )
    geometric_hfov = calculate_flat_screen_horizontal_fov_deg(
        args.screen_width, args.viewing_distance
    )
    camera_local = selected_camera_local(eye, state, args)
    return [
        "PRIUS STEERING-WHEEL CENTER CALIBRATION",
        f"STATUS: {'CONFIRMED' if state.confirmed else 'UNRESOLVED - PRESS ENTER/P TO CONFIRM'}",
        f"Source: {state.source}",
        (
            f"Wheel local X={state.wheel_local.x:+.4f} "
            f"Y={state.wheel_local.y:+.4f} Z={state.wheel_local.z:+.4f} m"
        ),
        (
            f"Wheel world X={wheel_world.x:+.4f} "
            f"Y={wheel_world.y:+.4f} Z={wheel_world.z:+.4f} m"
        ),
        f"Eye local   X={eye.x:+.4f} Y={eye.y:+.4f} Z={eye.z:+.4f} m",
        f"Monitor     X={monitor.x:+.4f} Y={monitor.y:+.4f} Z={monitor.z:+.4f} m",
        f"CAMERA VIEW={state.camera_view.upper()} at ({camera_local.x:+.4f}, {camera_local.y:+.4f}, {camera_local.z:+.4f})",
        f"Wheel-eye distance={distance * 1000.0:.1f} mm",
        f"Step={state.step_m * 1000.0:.0f} mm   camera pitch={state.pitch_deg:+.2f} deg",
        "W/S: +X/-X   A/D: -Y/+Y   R/F: +Z/-Z",
        "1: 10 mm   2: 5 mm   3: 1 mm",
        "PageUp/PageDown: camera pitch only",
        "M: toggle DRIVER EYE / MONITOR CENTER position",
        "Enter/P: confirm and save   Esc: exit",
        f"Mode={'DEBUG GEOMETRY' if args.debug_geometry else 'CLEAN (no debug primitives)'}",
        f"Flat baseline HFOV={geometric_hfov:.3f} deg (1500R excluded)",
    ]


def pedal_command(axis: float) -> float:
    """T300RS pedal axis (+1 released, -1 floored) to a 0..1 command."""
    value = 1.6 + (2.05 * math.log10(-0.7 * axis + 1.4) - 1.2) / 0.92
    return min(1.0, max(0.0, value))


def wheel_axis_to_steer(axis: float) -> float:
    """T300RS axis to a CARLA steer command through WHEEL_STEERING_RATIO."""
    axis = max(-1.0, min(1.0, axis))
    if abs(axis) <= WHEEL_STEER_DEADZONE:
        return 0.0
    road_wheel_deg = axis * WHEEL_RANGE_DEG * 0.5 / WHEEL_STEERING_RATIO
    return max(-1.0, min(1.0, road_wheel_deg / PRIUS_MAX_ROAD_WHEEL_DEG))


def prius_creep(control: Any, speed_mps: float) -> Any:
    """Hybrid creep in D/R: a small drive below walking pace that the brake can hold."""
    if speed_mps < PRIUS_CREEP_SPEED_MPS and not control.hand_brake:
        control.throttle = max(control.throttle, PRIUS_CREEP_THROTTLE)
    return control


def apply_prius_gen3_physics(vehicle: Any) -> None:
    physics = vehicle.get_physics_control()
    radius_m = physics.wheels[0].radius / 100.0
    # One fixed ratio so max_rpm lands on the governed top speed; the torque curve
    # then gives a wheel force of min(max force, power / speed), like the e-CVT.
    top_wheel_rpm = PRIUS_TOP_SPEED_KMH / 3.6 / (2.0 * math.pi * radius_m) * 60.0
    physics.forward_gears = [carla.GearPhysicsControl(ratio=1.0, down_ratio=0.0, up_ratio=0.0)]
    physics.final_ratio = physics.max_rpm / top_wheel_rpm
    torque_cap = PRIUS_MAX_DRIVE_FORCE_N * radius_m / physics.final_ratio
    rpm_per_rad_s = 60.0 / (2.0 * math.pi)
    knee_rpm = PRIUS_WHEEL_POWER_W / torque_cap * rpm_per_rad_s
    curve = [carla.Vector2D(0.0, torque_cap)]
    for i in range(6):
        rpm = knee_rpm + (physics.max_rpm - knee_rpm) * i / 5.0
        curve.append(carla.Vector2D(rpm, PRIUS_WHEEL_POWER_W / rpm * rpm_per_rad_s))
    physics.torque_curve = curve
    physics.mass = PRIUS_MASS_KG
    physics.drag_coefficient = PRIUS_DRAG_COEFFICIENT
    # The ratio is fixed on the real car (no speed-sensitive steering gear).
    physics.steering_curve = [carla.Vector2D(0.0, 1.0), carla.Vector2D(300.0, 1.0)]
    wheels = physics.wheels
    for wheel in wheels:
        front = wheel.max_steer_angle > 0.0
        if front:
            wheel.max_steer_angle = PRIUS_MAX_ROAD_WHEEL_DEG
        # Front-biased split, as on a front-heavy FWD hatch.
        wheel.max_brake_torque = PRIUS_BRAKE_TORQUE_NM * (1.0 if front else 0.5)
    physics.wheels = wheels
    vehicle.apply_physics_control(physics)


def wheel_vehicle_control(joystick: Any, reverse: bool, speed_mps: float) -> Any:
    control = carla.VehicleControl(
        throttle=pedal_command(joystick.get_axis(WHEEL_THROTTLE_AXIS)),
        brake=pedal_command(joystick.get_axis(WHEEL_BRAKE_AXIS)),
        steer=wheel_axis_to_steer(joystick.get_axis(WHEEL_STEER_AXIS)),
        hand_brake=bool(joystick.get_button(WHEEL_HANDBRAKE_BUTTON)),
        reverse=reverse,
    )
    return prius_creep(control, speed_mps)


def run_pygame_ui(
    state: CalibrationState,
    world: Any,
    vehicle: Any,
    camera: Optional[Any],
    frame_buffer: CameraFrameBuffer,
    args: argparse.Namespace,
    client_version: str,
    server_version: str,
) -> None:
    if pygame is None:
        raise RuntimeError("pygame is not installed")
    pygame.init()
    display_count = pygame.display.get_num_displays()
    if not 0 <= args.display_index < display_count:
        raise RuntimeError(
            f"--display-index {args.display_index} is invalid; "
            f"SDL reports {display_count} display(s)"
        )
    if args.fullscreen:
        desktop_sizes = pygame.display.get_desktop_sizes()
        screen_size = desktop_sizes[args.display_index]
        screen = pygame.display.set_mode(
            screen_size,
            pygame.FULLSCREEN,
            display=args.display_index,
        )
        pygame.mouse.set_visible(False)
        print(
            f"Fullscreen enabled on SDL display {args.display_index}: "
            f"{screen_size[0]}x{screen_size[1]}"
        )
    else:
        width = args.preview_width if args.preview_eye_camera else 920
        height = args.preview_height if args.preview_eye_camera else 520
        screen = pygame.display.set_mode(
            (width, height),
            display=args.display_index,
        )
    pygame.display.set_caption("Prius wheel-based driver-eye calibration")
    font = pygame.font.Font(None, 27)
    clock = pygame.time.Clock()
    pygame.key.set_repeat(250, 55)
    joystick = None
    reverse = False
    if args.wheel_drive:
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("--wheel-drive: no joystick/steering wheel found")
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        # The calibration spawn freezes the Prius; driving needs physics back.
        vehicle.set_enable_gravity(True)
        vehicle.set_simulate_physics(True)
        apply_prius_gen3_physics(vehicle)
        print(f"Wheel drive enabled: {joystick.get_name()}")
    eye = calculate_driver_eye(
        state.wheel_local, args.rearward, args.lateral_offset, args.upward
    )
    start = time.monotonic()
    last_debug = 0.0
    try:
        running = True
        while running:
            now = time.monotonic()
            if args.run_seconds > 0.0 and now - start >= args.run_seconds:
                break
            changed = False
            pitch_changed = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif (
                    event.type == pygame.JOYBUTTONDOWN
                    and event.button == WHEEL_REVERSE_BUTTON
                ):
                    reverse = not reverse
                    print(f"Gear = {'R' if reverse else 'D'}")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_p):
                        try:
                            confirm_manual_cursor(state)
                        except ValueError as exc:
                            print(f"REFUSED: {exc}")
                            continue
                        eye = print_measurement_report(state, vehicle, args)
                        save_calibration(
                            args.calibration_file,
                            state,
                            eye,
                            args,
                            client_version,
                            server_version,
                        )
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                        state.step_m = {
                            pygame.K_1: 0.010,
                            pygame.K_2: 0.005,
                            pygame.K_3: 0.001,
                        }[event.key]
                    elif event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                        state.pitch_deg += args.pitch_step * (
                            1.0 if event.key == pygame.K_PAGEUP else -1.0
                        )
                        pitch_changed = True
                    elif event.key == pygame.K_m:
                        view = toggle_camera_view(state)
                        camera_local = selected_camera_local(eye, state, args)
                        if camera is not None:
                            update_eye_camera(camera, camera_local, state, args)
                        print(
                            f"CAMERA VIEW = {view.upper()}: "
                            f"x={camera_local.x:.6f}, y={camera_local.y:.6f}, "
                            f"z={camera_local.z:.6f} m"
                        )
                    else:
                        keys = {
                            pygame.K_w: "w",
                            pygame.K_s: "s",
                            pygame.K_a: "a",
                            pygame.K_d: "d",
                            pygame.K_r: "r",
                            pygame.K_f: "f",
                        }
                        key = keys.get(event.key)
                        if key is not None:
                            changed = adjust_state(state, key) or changed
            if joystick is not None:
                velocity = vehicle.get_velocity()
                speed_mps = math.hypot(velocity.x, velocity.y)
                vehicle.apply_control(wheel_vehicle_control(joystick, reverse, speed_mps))
            if changed or pitch_changed:
                eye = _refresh(state, world, vehicle, camera, args)
            if now - last_debug >= max(0.1, args.debug_life_time * 0.5):
                draw_debug_geometry(world, vehicle, state, args)
                last_debug = now

            screen.fill((15, 17, 22))
            if args.preview_eye_camera:
                surface = frame_buffer.pygame_surface()
                if surface is not None:
                    if surface.get_size() != screen.get_size():
                        surface = pygame.transform.smoothscale(surface, screen.get_size())
                    screen.blit(surface, (0, 0))
                    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 90))
                    screen.blit(overlay, (0, 0))
            y = 18
            wheel_world = vehicle_local_to_world(
                vehicle.get_transform(), state.wheel_local
            )
            for line in _pygame_lines(state, eye, wheel_world, args):
                color = (255, 220, 80) if line.startswith("STATUS") else (245, 245, 245)
                screen.blit(font.render(line, True, color), (18, y))
                y += 29
            pygame.display.flip()
            clock.tick(args.ui_fps)
    finally:
        pygame.mouse.set_visible(True)
        pygame.quit()


def cleanup(camera: Optional[Any], vehicle: Optional[Any]) -> None:
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
    if vehicle is not None:
        try:
            if vehicle.is_alive:
                vehicle.destroy()
        except RuntimeError as exc:
            print(f"WARNING: Prius cleanup failed: {exc}", file=sys.stderr)


def validate_api() -> None:
    if carla is None:
        raise RuntimeError(f"Could not import carla: {CARLA_IMPORT_ERROR}")
    required = {
        "carla.Actor.set_simulate_physics": hasattr(carla.Actor, "set_simulate_physics"),
        "carla.Actor.set_enable_gravity": hasattr(carla.Actor, "set_enable_gravity"),
        "carla.Transform.transform": hasattr(carla.Transform, "transform"),
        "carla.Transform.inverse_transform": hasattr(carla.Transform, "inverse_transform"),
        "carla.AttachmentType.Rigid": hasattr(carla.AttachmentType, "Rigid"),
    }
    missing = [name for name, available in required.items() if not available]
    if missing:
        raise RuntimeError("CARLA Python API is missing: " + ", ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a vehicle-local Prius driver eye from the steering-wheel "
            "rotation-axis centre."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--max-spawn-attempts", type=int, default=40)
    parser.add_argument("--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE)
    parser.add_argument("--api-audit-output", type=Path, default=DEFAULT_API_AUDIT_FILE)
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument(
        "--initial-wheel",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Unconfirmed vehicle-local cursor seed supplied by the user.",
    )
    initial_group.add_argument(
        "--wheel-center",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Explicit user-measured vehicle-local wheel centre; treated as confirmed.",
    )
    parser.add_argument(
        "--ignore-saved-calibration",
        action="store_true",
        help="Do not load an existing confirmed calibration JSON.",
    )
    parser.add_argument(
        "--accept-api-candidate",
        action="store_true",
        help="Accept a unique semantic API pivot without pressing Enter.",
    )
    parser.add_argument("--rearward", type=float, default=RIG_REARWARD_M)
    parser.add_argument("--lateral-offset", type=float, default=DEFAULT_LATERAL_M)
    parser.add_argument("--upward", type=float, default=RIG_UPWARD_M)
    parser.add_argument(
        "--screen-width",
        type=float,
        default=MONITOR_WIDTH_M,
        help="Physical flat-display width used for geometry and automatic HFOV.",
    )
    parser.add_argument(
        "--viewing-distance",
        type=float,
        default=MONITOR_DISTANCE_M,
        help="Eye-to-monitor distance along vehicle-local +X.",
    )
    parser.add_argument("--monitor-lateral-offset", type=float, default=0.0)
    parser.add_argument("--monitor-vertical-offset", type=float, default=0.0)
    parser.add_argument("--preview-eye-camera", action="store_true")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--pitch-step", type=float, default=0.25)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Camera horizontal FOV; default is calculated from width/distance.",
    )
    parser.add_argument("--preview-width", type=int, default=1920)
    parser.add_argument("--preview-height", type=int, default=1080)
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Use exclusive fullscreen on the selected SDL display.",
    )
    parser.add_argument(
        "--display-index",
        type=int,
        default=0,
        help="SDL/Pygame display index used for the preview window.",
    )
    parser.add_argument(
        "--wheel-drive",
        action="store_true",
        help="Drive the Prius with a T300RS wheel (button 4 toggles reverse).",
    )
    parser.add_argument("--ui-fps", type=int, default=30)
    parser.add_argument("--text-ui", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--run-seconds", type=float, default=0.0)
    parser.add_argument("--auto-save", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--debug-geometry",
        action="store_true",
        help="Draw dim wheel/eye/axes/monitor validation primitives.",
    )
    mode_group.add_argument(
        "--clean-preview",
        action="store_true",
        help="Explicit clean mode (also the default): create no debug primitives.",
    )
    parser.add_argument("--debug-life-time", type=float, default=1.0)
    parser.add_argument("--debug-point-size", type=float, default=0.07)
    parser.add_argument("--marker-axis-length", type=float, default=0.15)
    parser.add_argument("--forward-line-length", type=float, default=3.0)
    parser.add_argument("--monitor-line-thickness", type=float, default=0.004)
    parser.add_argument("--monitor-center-point-size", type=float, default=0.025)
    parser.add_argument(
        "--gaze-line-thickness",
        type=float,
        default=0.006,
        help="Thin low-luminance gaze guide thickness in metres.",
    )
    parser.add_argument(
        "--guide-line-thickness",
        type=float,
        default=0.006,
        help="Wheel-to-eye guide thickness in metres.",
    )
    parser.add_argument(
        "--gaze-line-gray",
        type=int,
        default=70,
        help="Gaze-line grayscale intensity, 0 (black) to 255 (white).",
    )
    parser.add_argument(
        "--hide-gaze-line",
        action="store_true",
        help="Do not draw the eye-to-forward gaze guide.",
    )
    parser.add_argument(
        "--hide-gaze-label",
        action="store_true",
        help="Hide only the gaze-line text label.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout <= 0.0 or args.max_spawn_attempts < 1:
        raise ValueError("timeout and spawn attempt count must be positive")
    if args.rearward < 0.0 or args.upward < 0.0:
        raise ValueError("rearward/upward magnitudes must be non-negative")
    if args.fov is None:
        args.fov = calculate_flat_screen_horizontal_fov_deg(
            args.screen_width, args.viewing_distance
        )
    if not all(math.isfinite(value) for value in (
        args.rearward, args.lateral_offset, args.upward, args.pitch,
        args.pitch_step, args.yaw, args.roll, args.fov, args.screen_width,
        args.viewing_distance, args.monitor_lateral_offset,
        args.monitor_vertical_offset,
    )):
        raise ValueError("numeric camera/offset arguments must be finite")
    if not (1.0 <= args.fov <= 179.0):
        raise ValueError("--fov must be between 1 and 179 degrees")
    if args.pitch_step <= 0.0:
        raise ValueError("--pitch-step must be positive")
    if args.screen_width <= 0.0 or args.viewing_distance <= 0.0:
        raise ValueError("--screen-width and --viewing-distance must be positive")
    if args.preview_width < 64 or args.preview_height < 64 or args.ui_fps < 1:
        raise ValueError("preview dimensions/FPS are invalid")
    if args.display_index < 0:
        raise ValueError("--display-index must be zero or greater")
    if args.run_seconds < 0.0:
        raise ValueError("--run-seconds must be non-negative")
    if any(
        value <= 0.0
        for value in (
            args.debug_life_time,
            args.debug_point_size,
            args.marker_axis_length,
            args.forward_line_length,
            args.gaze_line_thickness,
            args.guide_line_thickness,
            args.monitor_line_thickness,
            args.monitor_center_point_size,
        )
    ):
        raise ValueError("debug sizes and lifetimes must be positive")
    if not 0 <= args.gaze_line_gray <= 255:
        raise ValueError("--gaze-line-gray must be between 0 and 255")
    if args.non_interactive and args.run_seconds == 0.0:
        args.run_seconds = 1.0


def choose_initial_state(
    args: argparse.Namespace,
    api_candidate: Optional[ApiCandidate],
) -> CalibrationState:
    if args.wheel_center is not None:
        return CalibrationState(
            Point3(*[float(value) for value in args.wheel_center]),
            "user_cli_measured",
            True,
            0.005,
            args.pitch,
        )
    if args.initial_wheel is not None:
        return CalibrationState(
            Point3(*[float(value) for value in args.initial_wheel]),
            "user_cursor_seed_unconfirmed",
            False,
            0.005,
            args.pitch,
        )
    if not args.ignore_saved_calibration:
        loaded = load_calibration(args.calibration_file)
        if loaded is not None:
            point, source = loaded
            return CalibrationState(point, source, True, 0.005, args.pitch)
    if api_candidate is not None:
        return CalibrationState(
            api_candidate.vehicle_local,
            f"api_{api_candidate.kind}_pivot:{api_candidate.name}",
            bool(args.accept_api_candidate),
            0.005,
            args.pitch,
        )
    return CalibrationState(
        Point3(0.0, 0.0, 0.0),
        "vehicle_origin_cursor_not_a_wheel_measurement",
        False,
        0.005,
        args.pitch,
    )


def run(args: argparse.Namespace) -> int:
    validate_api()
    validate_args(args)
    print("CARLA PRIUS STEERING-WHEEL REFERENCE AUDIT")
    print("Primary reference: steering-wheel rotation-axis centre")
    print("Excluded: seat, walker, bounding box, and previous demo-camera values\n")
    print(
        "Flat-screen horizontal FOV baseline = "
        f"{calculate_flat_screen_horizontal_fov_deg(args.screen_width, args.viewing_distance):.6f} deg"
    )
    print("1500R monitor curvature is intentionally not included.\n")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    client_version = str(client.get_client_version())
    server_version = str(client.get_server_version())
    world = client.get_world()
    print(f"CARLA client/server: {client_version} / {server_version}")
    print(f"World: {world.get_map().name}")

    vehicle = None
    camera = None
    frame_buffer = CameraFrameBuffer()
    try:
        vehicle = spawn_fixed_prius(world, args)
        print(f"Spawned and fixed: {vehicle.type_id} actor_id={vehicle.id}")
        candidates, _ = inspect_steering_wheel_api(
            vehicle,
            client_version,
            server_version,
            args.api_audit_output,
        )
        api_candidate = select_api_candidate(candidates)
        print(f"API audit saved: {args.api_audit_output}")
        if api_candidate is None:
            print(
                "API RESULT: steering-wheel rotation centre is not uniquely exposed. "
                "Interactive calibration fallback is active."
            )
        else:
            print(
                f"API CANDIDATE: {api_candidate.kind} '{api_candidate.name}' at "
                f"({api_candidate.vehicle_local.x:.6f}, "
                f"{api_candidate.vehicle_local.y:.6f}, "
                f"{api_candidate.vehicle_local.z:.6f}) vehicle-local m"
            )
            print("Its asset pivot still requires visual confirmation unless explicitly accepted.")

        state = choose_initial_state(args, api_candidate)
        if not state.confirmed:
            print(
                "WARNING: current point is an unresolved calibration cursor, not a "
                "steering-wheel centre measurement."
            )
        eye = print_measurement_report(state, vehicle, args)
        place_spectator(world, vehicle, state.wheel_local)
        draw_debug_geometry(world, vehicle, state, args)

        if args.preview_eye_camera:
            if state.source == "vehicle_origin_cursor_not_a_wheel_measurement":
                raise RuntimeError(
                    "--preview-eye-camera requires saved calibration, --wheel-center, "
                    "--initial-wheel, or a semantic API candidate"
                )
            camera_local = selected_camera_local(eye, state, args)
            camera = spawn_eye_camera(
                world, vehicle, camera_local, state, args, frame_buffer
            )
            print(
                "Rigid RGB camera spawned at DRIVER EYE. Press M to toggle its position "
                "between DRIVER EYE and VIRTUAL MONITOR CENTER; "
                "pitch/yaw/roll/FOV remain unchanged."
            )

        if args.auto_save:
            if not state.confirmed:
                raise RuntimeError(
                    "--auto-save refuses an unresolved cursor; confirm interactively, "
                    "use --wheel-center, load saved calibration, or accept an API candidate"
                )
            save_calibration(
                args.calibration_file,
                state,
                eye,
                args,
                client_version,
                server_version,
            )

        if args.non_interactive:
            end = time.monotonic() + args.run_seconds
            while time.monotonic() < end:
                draw_debug_geometry(world, vehicle, state, args)
                advance_world(world, 1, args.timeout)
        elif args.text_ui:
            run_text_ui(
                state,
                world,
                vehicle,
                camera,
                args,
                client_version,
                server_version,
            )
        else:
            try:
                run_pygame_ui(
                    state,
                    world,
                    vehicle,
                    camera,
                    frame_buffer,
                    args,
                    client_version,
                    server_version,
                )
            except (pygame.error if pygame is not None else RuntimeError) as exc:
                print(f"WARNING: Pygame unavailable ({exc}); switching to text UI")
                run_text_ui(
                    state,
                    world,
                    vehicle,
                    camera,
                    args,
                    client_version,
                    server_version,
                )

        final_eye = print_measurement_report(state, vehicle, args)
        if state.confirmed:
            print("Final wheel-centre status: CONFIRMED")
            print(
                f"Final driver-eye local: x={final_eye.x:.6f}, "
                f"y={final_eye.y:.6f}, z={final_eye.z:.6f} m"
            )
        else:
            print("Final wheel-centre status: UNRESOLVED; no final eye is claimed")
        return 0
    finally:
        cleanup(camera, vehicle)
        print("Cleanup complete: created camera/Prius actors destroyed.")
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


if __name__ == "__main__":
    raise SystemExit(main())
