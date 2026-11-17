# Literature basis for the CARLA seated-driver pose

Status: **RESEARCH DATA VERIFIED; CARLA BONE-AXIS VALIDATION NOT YET RUN**

This document records the evidence used to design a seated-driver pose for
`walker.pedestrian.0022` in `vehicle.toyota.prius`.  It does **not** establish
that this CARLA walker is a Korean female P50 mannequin.  The intended claim is
limited to “CARLA adult-female representative geometry candidate used for
driver-eye calibration.”

The published angles below are anatomical or landmark-based angles.  They are
not CARLA bone Euler rotations.  A pose JSON may be labelled
`literature_pose_validated=true` only after the live CARLA hierarchy has been
dumped, the flexion axis/sign for each required bone has been confirmed, and
the resulting landmark angles have been remeasured.

For the live CARLA remeasurement, the script uses explicit skeleton proxies:
the bilateral thigh roots approximate hip joint locations, clavicle/shoulder
bones approximate shoulder joint locations, lower-leg roots approximate knee
locations, foot roots approximate ankle locations, and foot-to-toe directions
approximate anatomical foot planes.  These rig points are not guaranteed to be
clinical joint centres.  That proxy error is retained as a limitation even
after the CARLA axis test passes.

## Primary posture dataset selected for the nominal targets

### Source

- Gyouhyung Kyung (2008), *An Integrated Human Factors Approach to Design and
  Evaluation of the Driver Workspace and Interface: Driver Perceptions,
  Behaviors, and Objective Measures*, PhD dissertation, Virginia Tech,
  especially Chapter 7, Tables 29–31.
- Original PDF: <https://vtechworks.lib.vt.edu/bitstreams/ed9034a2-405c-44bb-b179-d0bcf7bcf5af/download>

### Population and vehicle conditions

- 38 licensed drivers: 18 male and 20 female.
- Age groups were 20–35 and 60+ years; stature groups covered below 165 cm,
  middle stature, and above 175 cm. Grand mean stature was 170.0 cm.
- Sedan and SUV packages were studied in adjustable laboratory rigs and real
  vehicles. Participants drove for 20 minutes, adjusted seat and steering wheel
  to support their preferred posture, and then body landmarks were measured.
- The sedan results contain 113 posture observations (38 participants × three
  sedan sessions, with one missing observation).

### Angle definitions

- Torso: included direction from hip joint centre to shoulder joint centre,
  referenced to vertical in the sagittal plane.
- Hip: included angle at the hip between shoulder and knee joint centres.
- Knee: included angle at the knee between hip and ankle joint centres.
- Ankle: included angle at the ankle between the knee direction and either the
  anatomical foot plane or shoe plane.
- Shoulder and elbow: included angles between adjacent body segments.
- Neck: lower-neck to upper-neck segment referenced to vertical. It is not a
  head/Frankfort-plane angle.

### Reported sedan values and use

| Variable | Unfiltered range | Mean (SD) | Comfort-filtered MCMD groups | Nominal target used by the calibration solver | How it is used |
|---|---:|---:|---:|---:|---|
| Neck segment | −30–29° | −3.1° (11.4°) | 1–27° | −3.1° is retained as a reported reference only | Neck-line check; not called head pitch |
| Left shoulder | −1–59° | 29.6° (14.8°) | 1–29°, 32–58° | not applied in the first validated pose | Arms are lower priority and need a Prius steering-wheel target |
| Right shoulder | 2–64° | 28.9° (14.9°) | 3–26°, 35–59° | not applied in the first validated pose | Same limitation |
| Left elbow | 78–177° | 125.8° (21.1°) | 85–120°, 146–165° | not applied in the first validated pose | Same limitation |
| Right elbow | 78–179° | 120.6° (20.9°) | 85–108°, 133–167° | not applied in the first validated pose | Same limitation |
| Torso | 16–43° | 29.7° (5.6°) | 18–26°, 32–43° | **23.8°** | Reed OPM mean below lies inside Kyung MCMD group 1; avoids choosing the gap between comfort clusters |
| Left hip | 78–128° | 98.2° (9.7°) | 79–87°, 107–118° | **98.2° provisional landmark target** | Included-angle solver; must report that the mean lies between MCMD clusters |
| Right hip | 80–128° | 103.8° (9.9°) | 83–92°, 112–123° | **103.8° provisional landmark target** | Included-angle solver; same limitation |
| Left knee | 81–150° | 109.3° (12.5°) | 84–91°, 118–129° | **109.3° provisional landmark target** | Included-angle solver |
| Right knee | 89–152° | 120.4° (11.4°) | 93–110°, 123–142° | **120.4° provisional landmark target** | Right/pedal-side included-angle solver |
| Left ankle, foot plane | 75–130° | 96.4° (9.9°) | 82–88°, 92–123° | **96.4° provisional landmark target** | Anatomical foot-bone/toe direction, not shoe sole |
| Right ankle, foot plane | 76–132° | 95.5° (10.4°) | 77–91°, 108–112° | **95.5° provisional landmark target** | Same limitation; the mean is between MCMD groups |

“Provisional landmark target” means that the value is literature-backed but is
not yet a usable CARLA rotation.  It becomes usable only when the live solver
can demonstrate that the corresponding measured included angle converges to
the target using a verified bone axis.

## Multivariate driver-posture cross-check

### Source

- Matthew P. Reed, Miriam A. Manary, Carol A. C. Flannagan, and Lawrence W.
  Schneider (2000), “Comparison of Methods for Predicting Automobile Driver
  Posture,” SAE Technical Paper 2000-01-2180.
- Author-hosted PDF: <https://mreed.umtri.umich.edu/mreed/pubs/Reed_2000-01-2180.pdf>

### Population and vehicle type

The optimization-prediction model (OPM) used automobile driver data gathered
over varied seat and package geometries. Its posture vector was knee, elbow,
and torso angle. The paper reports a population mean torso angle of 23.8° and
knee angle of 118.0°. Within-subject standard deviations were 2.8° and 7.9°,
respectively. The elbow mean is an anthropometry-dependent regression rather
than one universal value.

### How it is used

- **Torso target = 23.8° from vertical.** It is the OPM population mean and
  falls inside Kyung’s sedan comfort-filtered 18–26° group.
- Knee = 118° is a cross-check, not forced bilaterally, because Kyung measured
  substantial left/right differences and provides sedan-specific means.
- No universal elbow value is selected. Steering-wheel reach and mannequin
  proportions must be known before arm posture is solved.

### Limitations

The OPM correlations show that independently selecting marginal means can
produce a posture that is not representative of any one driver. Package
constraints must be honoured. Neither this paper nor the Kyung dissertation
defines CARLA rig rotations.

## Peer-reviewed package-dependent posture-model context

### Source

- Matthew P. Reed, Miriam A. Manary, Carol A. C. Flannagan, and Lawrence W.
  Schneider (2002), “A Statistical Method for Predicting Automobile Driving
  Posture,” *Human Factors*, 44(4), 557–568.
- Author-hosted paper: <https://mreed.umtri.umich.edu/mreed/pubs/Reed_2002.pdf>

### Population, vehicle conditions, and use

The model-development data covered 68 men and women in 18 vehicle-package and
seat conditions. Predictions were evaluated against 120 men and women in five
vehicles. The method couples regression models with data-guided inverse
kinematics and gives high priority to eye and hip locations.

This paper is used to justify retaining package constraints, bilateral
landmark geometry, and eye/hip outputs in the solver. It is **not** used as a
source of one universal CARLA joint vector. Its reported typical mean eye
prediction error below 10 mm is performance of that statistical model in its
validation vehicles; it is not an accuracy claim for this CARLA tool.

## Pelvis and lumbar posture evidence

### Source

- Matthew P. Reed and Lawrence W. Schneider (1996), “Lumbar Support in Auto
  Seats: Conclusions from a Study of Preferred Driving Posture,” SAE Technical
  Paper 960478. Author-hosted PDF:
  <https://mreed.umtri.umich.edu/mreed/pubs/Reed_19960478.pdf>

### Reported values

With a flat lumbar support, the preferred seated posture showed a mean pelvis
rotation of 55.4° rearward relative to each participant’s standing posture and
a sternum recline change of 6.2°; their difference represented 49.2° of net
spine flexion. A 45-mm lumbar support changed pelvis and sternum orientations
by about 3.7° in opposite directions.

### How it is used

These results establish that pelvis and spine changes must not be replaced by
one arbitrary whole-body “seat-back rotation.” The 55.4° pelvis change is
stored as a literature reference, but it is **not** applied directly to the
CARLA pelvis bone. The pelvis/spine chain must be calibrated from measured
landmark geometry after the CARLA axis test.

### Limitations

The study used a particular test seat and lumbar-support setup. Its pelvis
angle is a landmark/segment measure relative to standing, not an H-point,
SgRP, or CARLA Euler angle.

## Head and gaze

The reviewed driver datasets provide a neck-segment inclination but do not
provide a directly transferable CARLA head-bone pitch target for this exact
task. Consequently:

- `head_angle_deg` remains **N/A** in a pose result until a source with a
  compatible head-angle definition is selected and the CARLA head axis is
  verified.
- Eye position is measured from the two eye bones regardless of head target.
- Camera pitch/yaw/roll and FOV remain independent gaze/display parameters.
- A zero-degree camera pitch is a preview setting, not a literature-derived
  driver gaze.

## SAE/ISO concepts and their roles

### SAE J941 driver eyellipse / ISO 4513

- SAE J941 and ISO 4513 define statistical three-dimensional distributions of
  driver eye locations for visibility design. They do not prescribe skeleton
  joint angles.
- ISO 4513 describes the eyellipse centroid as a function of package geometry
  including SgRP, steering-wheel location, and cushion angle. The current CARLA
  eye midpoint may later be compared with an eyellipse, but J941 is not used to
  generate the pose.
- Sources: <https://saemobilus.sae.org/standards/j941_199706-motor-vehicle-drivers-eye-locations>
  and <https://www.iso.org/obp/ui/#iso:std:45845:en>.

### H-point versus human pelvis

- ISO 6549 defines the H-point as the torso/thigh pivot of the physical H-point
  machine. The actual human hip centre and a CARLA pelvis bone are related
  concepts but are not identical points.
- Source: <https://www.iso.org/standard/75286.html> and the publicly described
  ISO 6549 definitions.

### R-point / SgRP

- R-point/SgRP is the manufacturer’s unique design H-point and a vehicle-fixed
  packaging reference. It is not automatically available from the CARLA Prius
  asset and must not be inferred from the walker pelvis.
- ISO 20176 explicitly states that H-point devices audit seat reference points;
  they do not measure occupant capability or comfort.

### Driver-selected seat position

- SAE J1517 predicts distributions of selected horizontal H-point locations as
  functions of package geometry (notably H30), for specified populations.
- It locates seat travel/accommodation and does not provide a complete joint
  pose. Source: <https://saemobilus.sae.org/standards/j1517_200902-driver-selected-seat-position>.

### Preferred posture and seat-back angle

- Preferred posture is the posture selected by human participants under a
  defined package/task. Kyung’s results show vehicle class, venue, seat, and
  driver attributes can affect it.
- SAE A40/design torso or seat-back angle is a package/design measurement. It
  is not necessarily the orientation of a person’s anatomical trunk or the
  physical seatback surface.

### Steering-wheel reach and pedal reach

- These are kinematic constraints, not optional cosmetic adjustments. Reed’s
  driver-posture research found that changing steering-wheel/package geometry
  altered elbow and knee postures while torso posture changed less.
- The first CARLA literature pose therefore validates pelvis, torso, and legs
  before arms. Arms remain unvalidated until the Prius steering-wheel rim/grip
  target is measured.

## Selected nominal targets

| Target | Selected value |
|---|---:|
| Trunk inclination from vertical | 23.8° |
| Left/right hip included angles | 98.2° / 103.8° (provisional) |
| Left/right knee included angles | 109.3° / 120.4° (provisional) |
| Left/right ankle, anatomical foot plane | 96.4° / 95.5° (provisional) |
| Neck segment from vertical | −3.1° reference only |
| Pelvis rearward change from standing | 55.4° reference only |
| Head orientation | N/A |
| Shoulder and elbow pose | deferred pending Prius steering-wheel target |

Representative values are not claimed to describe Korean women. They are a
traceable starting vector for a CARLA sedan-driver pose. The left/right means
are retained rather than forcing symmetry. Where a mean falls outside the
comfort-filtered clusters, the limitation is explicit and the value must not be
labelled a comfort optimum.

## Validation gate

`validated_driver_seated_pose.json` must **not** be generated as validated
until all of the following are recorded from a live CARLA 0.9.16 session:

1. Full `walker.pedestrian.0022` hierarchy and parent-relative neutral
   transforms.
2. Required anatomical-to-bone mapping, including toe/ball direction for the
   ankle definition.
3. Positive and negative X/Y/Z rotation perturbations for each controlled bone.
4. The selected sagittal flexion axis and sign, confirmed from measured child
   displacement and visual debug output.
5. Remeasured torso, bilateral hip, knee, and ankle landmark angles within a
   declared tolerance of their targets.
6. Pelvis re-alignment to a user-supplied Prius-local seat target. This target
   is a geometry calibration point, **not** an asserted H-point/SgRP.

Until then the program must print `CARLA BONE-AXIS VALIDATION REQUIRED` and
must not claim a literature-validated seated eye point.
