# Toyota Prius steering-wheel centre determination in CARLA 0.9.16

## Conclusion

The stock `vehicle.toyota.prius` asset does not provide a documented CARLA
Python property named “steering-wheel centre”. Physics wheel metadata describes
the four road wheels and must not be used for the cockpit steering wheel.

The locally installed Python binding has project-specific actor-introspection
extensions:

- `get_component_names()` / `get_component_world_transform()`
- `get_bone_names()` / `get_bone_world_transforms()`
- `get_socket_names()` / `get_socket_world_transforms()`

These methods make direct extraction possible only if the spawned stock Prius
contains a semantically named steering-wheel component, bone, or socket whose
pivot is the rotation centre. The integrated tool queries the live actor and
accepts only names containing both `steer` and `wheel`. It converts the returned
world pivot to vehicle-local coordinates using the complete inverse vehicle
transform.

The original read-only stock-asset audit was recorded separately in the source
workspace. The relevant result was that the asset
contains only these skeletal bones:

- `Vehicle_Base`
- `Wheel_Front_Left`
- `Wheel_Front_Right`
- `Wheel_Rear_Left`
- `Wheel_Rear_Right`

It contains no cockpit steering-wheel bone. A separate custom blueprint in the
workspace has an independent `SteeringWheel` component, but its pivot belongs
to `vehicle.custom.prius_xw30_demo`, not `vehicle.toyota.prius`, and is therefore
not reused here.

If the live stock actor does not expose one unique semantic pivot, the result is
reported as “not uniquely extractable” and manual visual calibration is used.
The vehicle origin is only an unresolved cursor seed and cannot be saved without
moving it or supplying an explicit measurement.

## Physical rig mapping

CARLA vehicle-local axes are `+X forward`, `+Y right`, `+Z up`.

For a calibrated steering-wheel centre `W`:

```text
Eye.x = W.x - 0.500
Eye.y = W.y + lateral_offset
Eye.z = W.z + 0.210
```

The default lateral offset is zero and is configurable with
`--lateral-offset`. With zero lateral offset, the calculated distance is
`sqrt(0.500² + 0.210²) = 0.542310 m`. The rearward value comes from the seated
rig: seatback → monitor 0.910 m and seatback → eye 0.200 m (both at 1.055 m
height) give eye → monitor 0.710 m, minus wheel → monitor 0.210 m.

## Flat-display horizontal FOV baseline

```text
2 atan((0.6984 / 2) / 0.710) = 52.378778 degrees
```

This is a flat-screen pinhole baseline. The MSI MAG 32C6X 1500R curvature is
not included. Camera position, camera rotation, and FOV remain separate
parameters.

## Interactive controls

```text
W/S               local +X / -X
A/D               local -Y / +Y
R/F               local +Z / -Z
1                 10 mm step
2                  5 mm step
3                  1 mm step
PageUp/PageDown   camera pitch only
M                 toggle camera position: driver eye / virtual monitor centre
Enter or P        confirm and save
Esc               exit
```

The point to align is the geometric centre of the steering-wheel rotation axis,
not either edge of the rim.

## Virtual monitor geometry and clean preview

The RGB camera remains rigidly attached at the calculated driver-eye point.
The physical display is represented separately as a flat vehicle-local Y/Z
plane whose centre is calculated from the current eye point:

```text
monitor.x = eye.x + viewing_distance
monitor.y = eye.y + monitor_lateral_offset
monitor.z = eye.z + monitor_vertical_offset
```

The baseline values are 0.710 m viewing distance, 0.6984 m width, 16:9
aspect ratio, and zero lateral/vertical monitor offset. This gives a 0.39285 m
height and 52.378778 degrees flat-screen horizontal FOV.

The updated longitudinal viewing distance is the measured eye-to-wheel
distance (0.500 m) plus wheel-to-monitor distance (0.210 m). With the saved
eye at (0.315, -0.415, 1.130) m, the monitor centre is
(1.025, -0.415, 1.130) m. The saved wheel calibration JSON remains a historical
record: loading it uses only its wheel centre and source, not its old FOV or
monitor-distance metadata.

Automatic FOV from the updated geometry:

```bash
python3 prius_wheel_based_driver_eye.py \
  --preview-eye-camera --clean-preview --lateral-offset 0.0
```

To explicitly use the rounded value, add `--fov 52.38`.

CARLA `world.debug` primitives do not expose a material interface for setting
roughness or guaranteeing non-emissive rendering. Consequently, clean preview
is the default and creates no debug points, lines, axes, cursor, or monitor
rectangle. `--debug-geometry` explicitly enables thin, low-intensity validation
primitives for the wheel, eye, monitor rectangle, local axes, and connecting
lines.

With RGB preview active, pressing `M` reuses the same rigid camera sensor and
toggles only its vehicle-local position between the driver-eye point and the
virtual monitor centre. Camera pitch, yaw, roll, and FOV are left unchanged so
the two positions can be compared under identical projection settings. The
monitor-centre view is a diagnostic comparison and is not the calibrated
driver-eye perspective.
