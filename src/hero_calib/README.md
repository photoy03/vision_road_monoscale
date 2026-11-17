# CARLA Prius Driver-Eye Calibration

CARLA 0.9.16의 `vehicle.toyota.prius`에서 steering-wheel rotation centre를
기준으로 운전자 eye point와 virtual monitor geometry를 계산하고 확인하는
Python 도구입니다.

## Calibrated geometry

CARLA vehicle-local axes are `+X forward`, `+Y right`, `+Z up`.

| Point / parameter | Value |
|---|---:|
| Steering-wheel centre | `(0.815, -0.415, 0.920) m` |
| Wheel → eye offset | `(-0.500, 0.000, +0.210) m` |
| Driver eye | `(0.315, -0.415, 1.130) m` |
| Eye → monitor distance | `0.710 m` |
| Virtual monitor centre | `(1.025, -0.415, 1.130) m` |
| Display width / height | `0.6984 / 0.39285 m` |
| Calculated horizontal FOV | `52.378778°` |

Eye → monitor comes from the seated rig, measured at 1.055 m height:
seatback → monitor 0.910 m minus seatback → eye 0.200 m. The monitor stays
0.210 m ahead of the wheel centre, so eye → wheel is 0.500 m.

The FOV is calculated at runtime:

```text
HFOV = 2 atan((screen_width / 2) / viewing_distance)
```

The calibration belongs to the measured physical rig. It is not a universal
Toyota Prius or human anthropometry value.

## Requirements

- Ubuntu 22.04 or a compatible Linux environment
- CARLA 0.9.16 server running on `127.0.0.1:2000`
- CARLA 0.9.16 Python API available to the selected Python interpreter
- Python packages in `requirements.txt`

Install the ordinary Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install the CARLA Python API using the package distributed with the same CARLA
0.9.16 installation. Do not install a mismatched CARLA client version.

## Run

Clean FHD eye-camera preview:

```bash
python3 prius_wheel_based_driver_eye.py \
  --preview-eye-camera \
  --clean-preview \
  --lateral-offset 0.0
```

Fullscreen preview on SDL display 0:

```bash
./run_fullscreen.sh
```

For another monitor index:

```bash
DISPLAY_INDEX=1 ./run_fullscreen.sh
```

Drive the Prius with a Thrustmaster T300RS wheel (physics is re-enabled; button
4 toggles reverse, button 5 is the handbrake):

```bash
./run_fullscreen.sh --wheel-drive
```

Wheel drive also swaps CARLA's stock Prius physics for 3rd-generation (ZVW30)
values: 37.8 deg lock reached at the wheel's stop (14.27:1), 1380 kg, Cd 0.25, 0-100 km/h in
10.4 s, hybrid creep. Sources and measured response: `prius_gen3_physics.md`.

Geometry validation view:

```bash
python3 prius_wheel_based_driver_eye.py \
  --preview-eye-camera \
  --debug-geometry
```

The camera starts at the virtual-monitor centre. Press `M` to compare the
monitor-centre and driver-eye camera positions while retaining the same camera
rotation and FOV. Press `Esc` to exit.

## Publish this prepared repository privately

On the original Ubuntu desktop, run:

```bash
./publish_private_github.sh
```

The helper installs Ubuntu's `gh` package when necessary, opens GitHub's web
authentication flow, uses or creates `UnKaiFoo/HERO_Calib` as a private
repository, pushes `main`, and opens the resulting repository page.

## Controls

| Input | Action |
|---|---|
| `M` | Toggle driver eye / monitor centre camera position |
| `PageUp`, `PageDown` | Adjust camera pitch |
| `W/S`, `A/D`, `R/F` | Adjust local wheel cursor X/Y/Z |
| `1`, `2`, `3` | Select 10/5/1 mm calibration step |
| `Enter` or `P` | Confirm and save calibration |
| `Esc` | Exit |

## Files

- `prius_wheel_based_driver_eye.py`: main application
- `prius_steering_wheel_center.json`: confirmed rig calibration
- `prius_steering_wheel_center_method.md`: method and API findings
- `prius_steering_wheel_api_audit.json`: recorded actor API audit
- `prius_gen3_physics.md`: 3rd-generation Prius physics for `--wheel-drive`, sources and measurements
- `run_fullscreen.sh`: FHD fullscreen preview helper
- `publish_private_github.sh`: private GitHub install/auth/create/push helper
- `docs/`: seated-pose research and walker bone mapping
- `legacy/`: earlier mannequin-based tools and measurements retained for traceability

The main application resolves its JSON files relative to the script directory,
so the repository can be cloned under any user account. Some files in
`legacy/` preserve historical absolute output paths and may need explicit CLI
path arguments on another computer.

## CARLA API limitation

Standard CARLA does not document a direct cockpit steering-wheel-centre
property. The tool checks for semantic component, bone, and socket inspection
methods when available, then falls back to the confirmed manual calibration.
Debug primitives cannot guarantee a non-emissive material, so clean preview is
the default and draws no debug geometry.
