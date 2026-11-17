# Prius 3rd-generation physics for `--wheel-drive`

`--wheel-drive` replaces CARLA's stock `vehicle.toyota.prius` physics with
values for the 3rd-generation Prius (ZVW30, 2010, 15in 195/65R15). Values that
come from specs are applied as is; values CARLA cannot express directly are
tuned until the measured CARLA response matches the published figure.

## Spec values

| Item | Value | Source |
|---|---|---|
| Curb weight | 1380 kg | Toyota Canada 2010 Prius brochure, spec page |
| Drag coefficient | 0.25 | same brochure |
| Turning circle | 10.4 m | same brochure |
| Wheelbase | 2700 mm | same brochure |
| Hybrid system net power | 100 kW (134 hp) | same brochure |
| Steering ratio | 17.6:1 (15in) / 14.6:1 (17in) | 2010 European spec sheet |
| Lock-to-lock | 3.7 turns (15in) / 2.8 (17in) | same sheet |
| 0-100 km/h | 10.4 s | auto-data.net, ZVW30 1.8 |
| Top speed | 180 km/h | auto-data.net |
| 60 mph-0 | 118 ft (36.0 m) / 133 ft (40.5 m) | Edmunds / Consumer Reports, both quoted on PriusChat |

Cross-check of the steering: 3.7 turns at 17.6:1 gives 37.8 deg at the road
wheel. The brochure turning circle and wheelbase give 36.3 deg at the axle
centre (bicycle model). An aftermarket 2010-2015 rack lists 3.63 turns.

## What is applied

| CARLA field | Stock | Gen3 | Basis |
|---|---|---|---|
| `mass` | 1375 | 1380 | spec |
| `drag_coefficient` | 0.30 | 0.25 | spec |
| front `max_steer_angle` | 70 deg | 37.8 deg | 3.7 turns / 17.6 |
| `steering_curve` | 1.0 -> 0.7 at 120 km/h | flat 1.0 | the real ratio is fixed |
| wheel -> road wheel | wheel end = CARLA 70 deg | wheel deg / 14.27 | real lock at the T300 stop (spec 17.6 cannot reach it) |
| gears, `final_ratio` | 1 gear, 4.17 | 1 gear, `max_rpm` = 180 km/h | governed top speed |
| torque curve | flat 450 Nm then falls | wheel force min(5000 N, P/v) | e-CVT: constant power |
| P in the curve | | 117 kW | tuned to 0-100 = 10.4 s |
| brake torque | 800 all | 1500 front / 750 rear | tuned to 60 mph-0 in the 36-40.5 m band |
| creep | none | throttle 0.12 below 7 km/h | hybrid creeps in D/R |

P is above the 100 kW spec because CARLA's rolling, damping and drag losses
eat about a third of it; 100 kW gives 11.2 s.

## Measured in CARLA

Town05_Opt, spawn 62, synchronous 0.01 s
(`prius_bench.py` in the session scratchpad, not in this repository).

| | Stock | Gen3 | Target |
|---|---|---|---|
| 0-100 km/h | 9.96 s | 10.40 s | 10.4 s |
| 60 mph-0 | 46.2 m | 37.2 m | 36.0-40.5 m |
| 100-0 km/h | 49.1 m | 39.6 m | |
| Turning circle, full lock | 7.39 m | 11.12 m | 10.4 m |
| Turning circle, T300 at its stop | 7.39 m | 11.12 m | 10.4 m |
| Creep, no pedals, 8 s | 0 km/h | 4.9 km/h | |

## Limits

- **CARLA's mesh keeps its own geometry.** The asset has a 2.82 m wheelbase
  and 0.37 m wheel radius against 2.70 m and 0.317 m. Scaling 10.4 m by
  2.82/2.70 gives 10.9 m, so most of the 11.12 m comes from the asset.
- **The wheel ratio is 14.27:1, not 17.6:1.** The T300RS turns ±540 deg; the
  real car needs ±666 deg (3.7 turns). Kept at 17.6:1 the wheel stopped at
  30.7 deg (81% of lock, 11.73 m circle), so the ratio was shortened to put
  the real 37.8 deg lock at the wheel's stop. The wheel is 23% quicker than
  the car; the lock and the turning circle are the car's.
- **Top speed was not reached in a run.** The longest Town05 straight is
  275 m. At 145 km/h full throttle still gives +0.68 m/s², so the limit comes
  from `max_rpm` = 180 km/h.
- **Creep has no source.** The 7 km/h threshold and 0.12 throttle are
  set by feel.
- 0-50 km/h (4.7 s) is set by the 5000 N launch force, which is also not
  from a spec.
