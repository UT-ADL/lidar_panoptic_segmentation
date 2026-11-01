# lidar_panoptic_segmentation

## Demo

A neural network that labels every point of a rotating lidar's scan in one pass: ground, static obstacle or dynamic
obstacle; object instances of the dynamic points; one of the 30 nuScenes classes; and, from the 4 previous scans, the
over-ground velocity of every dynamic object. In the lab's autonomy stack it replaces ground removal and clustering:
static points go to the collision checker as points, dynamic objects with their classes and velocities go to the
tracker. This repository holds the model as one ONNX file, an example scan from our car and one script to run both.

[Demo video: the model on our car in Tartu](https://youtu.be/r0lbQwu6Mmk?si=keMga1_NuFt5JODf)

![model output on the example scan](example.png)

The example scan (`example.pcd`, Tartu, 2026-04-22) seen from above and behind the car: ground grey, static obstacles
green, dynamic objects coloured by instance, dynamic points without an instance red, black arrows the velocity of
moving objects (1 s ahead). A bus approaches from ahead on the left; pedestrians and parked cars all around.

## Quick start

```bash
pip install -r requirements.txt
python segment.py example.pcd
```

prints (CPU; the object list abbreviated):

```
pseg_v6.1_fp16.onnx on CPUExecutionProvider: 47827 points, 4 previous scans
  ground 19632, static 9742, dynamic 6072, none 12381 (outside the grid, above max_height or isolated)
38 objects (dynamic points without an instance: 319):
    1  vehicle.car                            119 points  speed  0.0 m/s
    ...
   10  human.pedestrian.adult                  92 points  speed  1.3 m/s
    ...
   17  vehicle.bus.rigid                     3413 points  speed  7.0 m/s
    ...
   24  vehicle.truck                          342 points  speed 11.0 m/s
    ...
```

`segment.py` reads the previous scans `example_sweep_1.pcd` .. `example_sweep_4.pcd` (1 = newest) and their time lags
from `sweeps.json` next to the scan, and uses the CUDA provider of onnxruntime if it is available, else the CPU. It
reads PCD files with `DATA ascii` or `binary` and the fields x y z intensity. Options:

- `--lidar_height 2.11`: the lidar's height above the ground (m);
- `--no-sweeps`: without previous scans (segmentation stays, velocity is not usable);
- `--show`: interactive 3D view, `--plot out.png`: the same scene rendered offscreen to a PNG. Both need
  `pip install open3d` (tested with 0.19.0); the printed object list is their legend;
- `--out results.npz`: the per-point results (points, group, inst, native, velocity, class_names);
- `--cpu`: the CPU even if CUDA is available.

On a 16-core CPU (Python 3.9, onnxruntime 1.19) `segment.py` takes about 4 s for the example including model
loading, with 1.7 GB of memory. On a GPU a scan takes about 20 ms with TensorRT fp16 and about 0.23 s with
onnxruntime's CUDA provider (V100). `--plot` renders offscreen through EGL: it works on a desktop or a machine with an
NVIDIA driver, not on a headless machine without one (Open3D cannot initialise EGL there).

## Model

- **Input:** the current scan (x, y, z, raw intensity in the lidar frame) and the 4 previous scans moved into the
  current lidar frame, with their time lags.
- **Network:** a per-point MLP max-pooled into a 0.2 m bird's-eye-view grid over +-76.8 m (768 x 768 cells, 4 height
  slices), a residual CNN backbone with an FPN neck at 0.4 m, and heads for a BEV object-centre heatmap, per-point
  group, thing, offset to the object centre and classes, and a BEV velocity map on its own neck.
- **Decoding inside the graph:** heatmap peaks are object centres; dynamic thing points shifted by their predicted
  offset join the nearest centre within 2 m. A local ground height is estimated from the ground points, and points more
  than `max_height` above it (canopy) are dropped. Isolated static points within `filter_range` are dropped as noise.
  Obstacle points without an instance that move faster than 2 m/s next to an instance join it.
- **Output per point:** group (ground / static / dynamic / none), instance id (per scan, not tracked), nuScenes class,
  xy velocity of its object.
- **fp16:** `pseg_v6.1_fp16.onnx` (23 MB) is the deployed model `pseg_v6.1` with the network's weights and computation
  converted to fp16 by onnxruntime's float16 converter (`keep_io_types`); the in-graph pre- and post-processing stays
  fp32. On our test set through the deployed node code (onnxruntime, CUDA) it gives PQ 0.857, the same as the deployed
  TensorRT fp16 engine (the fp32 model: 0.861), and per-object velocity errors of 1.04 / 0.33 m/s for moving / static
  objects (fp32: 0.98 / 0.34). Against the fp32 model it agrees on 98.3% of the points' groups, 99.6% of their instance
  membership and 95.7% of their classes. On the CPU the fp16 model computes partly through casts and is not faster
  than fp32; its advantage is the file size (fp32: 42 MB).
- **Class names:** `pseg_v6.1_fp16.yaml` and the ONNX metadata key `class_names`.

The model was trained with the lab's training code, which is not public.

**Limitations:**

- The classes come from the nuScenes head; the other datasets' heads are in the network but not exported.
- The "unknown" style is used for every input; it was chosen on our test set, not tuned for other sensors.
- Trained on 32-64 beam rotating lidars at 10-20 Hz; other sensors (solid-state, much denser or sparser scans) are
  outside the training distribution. Points closer than 2.7 m were not seen at deployment.
- Instance ids are per scan; there is no temporal identity (use a tracker).
- The previous scans must be motion-compensated by the caller; wrong poses give wrong velocities.
- Long vehicles (buses, trucks with trailers) are sometimes split into two objects at long range.

## ONNX interface

Inputs:

| name | shape | dtype | meaning |
|---|---|---|---|
| `points` | (65536, 4) | float32 | the current scan: x, y, z (m, lidar frame: x forward, y left, z up, origin at the lidar), raw intensity (VLP-32C: 0-255). Unused rows: padding `(1e4, 0, 0, 0)`, a point far outside the grid. Points closer than 2.7 m should be removed (the deployed lidar driver's minimum range). |
| `sweeps` | (262144, 5) | float32 | the previous 4 scans, see [Multi-sweep input](#multi-sweep-input). Padding `(1e4, 0, 0, 0, 0)` |
| `max_height` | (1,) | float32 | non-ground points more than this above the local predicted ground get group -1 (deployed 2.5 m) |
| `lidar_height` | (1,) | float32 | the lidar's height above the ground in m (our car: 2.11) |
| `filter_range` | (1,) | float32 | within this range (m), static points with fewer than `min_static_points` static points in the surrounding 3 x 3 cells of 0.2 m are dropped as noise (deployed 38 m: beyond it a real obstacle hit by one VLP-32C beam can have fewer returns) |

Outputs (one row per row of `points`; padding rows give -1 / 0):

| name | shape | dtype | meaning |
|---|---|---|---|
| `group` | (65536,) | int32 | 0 ground, 1 static, 2 dynamic, -1 none (padding, outside the grid, above `max_height`, isolated static point) |
| `inst` | (65536,) | int32 | instance id 1..256 of dynamic points with an object centre, 0 none; ids are per scan. The deployed node groups dynamic points without an instance by connected 0.5 m cells. |
| `native` | (65536,) | int32 | index into the `class_names` metadata (nuScenes lidarseg classes) |
| `velocity` | (65536, 2) | float32 | over-ground xy velocity of the point's object, m/s, lidar frame axes |

Metadata (`metadata_props`):

| key | value | meaning |
|---|---|---|
| `sweeps` | `4` | number of previous scans |
| `sweep_points` | `262144` | rows of the `sweeps` input |
| `velocity` | `true` | the model has the `velocity` output |
| `moving_static_speed` | `2.0` | obstacle points without an instance faster than this (m/s) within 1 m of an instance join it |
| `moving_agree` | `1.5` | ... if their velocity is within this (m/s) of the instance's |
| `class_names` | JSON list of 30 names | `native` index -> nuScenes class: `animal`, `human.pedestrian.adult`, `.child`, `.construction_worker`, `.personal_mobility`, `.police_officer`, `.stroller`, `.wheelchair`, `movable_object.barrier`, `.debris`, `.pushable_pullable`, `.trafficcone`, `static_object.bicycle_rack`, `vehicle.bicycle`, `vehicle.bus.bendy`, `vehicle.bus.rigid`, `vehicle.car`, `vehicle.construction`, `vehicle.emergency.ambulance`, `.police`, `vehicle.motorcycle`, `vehicle.trailer`, `vehicle.truck`, `flat.driveable_surface`, `flat.other`, `flat.sidewalk`, `flat.terrain`, `static.manmade`, `static.other`, `static.vegetation` |
| `pseg` | JSON | the network configuration (`cfg`: grid x / y -76.8..76.8 m, z -1..3 m above the ground in 4 slices, voxel 0.2 m, output stride 2, sweeps 4, velocity map on its own neck) and the export constants: `height` 2.11 (lidar height at export), `intensity` `quantile:av2` (raw intensity mapped to [0, 1] with a quantile table of Argoverse 2's VLP-32C lidars), `embed` `unknown`, `head` `nuscenes`, `points` 65536, `ground_cell` 1.0 m, `min_static_points` 4, `dynamic_bias` 2.0 (added to the dynamic group logit), `stuff_fallback` "", `spray_height` 0.0, `spray_range` 30.0 (rain spray rule, off) |

## Styles and datasets

The model was trained on the union of four public lidar datasets. Each has its own class head, and all share the
ground / static / dynamic, thing, instance and velocity heads. A learned style embedding per dataset adapts the network
to each sensor and labelling convention, plus an "unknown" style trained on 10% of the frames. The export fixes the
embedding to "unknown" (the best on our own test set) and the class output to the nuScenes head, so the ONNX model has
no style or dataset input.

| dataset | what it contributed | licence (quoted from the terms) | URL |
|---|---|---|---|
| nuScenes with nuScenes-lidarseg and Panoptic nuScenes | Velodyne HDL-32E scans, 20 Hz; point classes and instances (the exported class head), sweeps, instance velocities | "Unless specifically labeled otherwise, these Datasets are provided to You under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License ("CC BY-NC-SA 4.0"), with the additional terms included herein." (nuScenes Terms of Use, 16 Nov 2021) | https://www.nuscenes.org/terms-of-use |
| SemanticKITTI (on KITTI odometry) | Velodyne HDL-64E scans, 10 Hz; point classes and instances | SemanticKITTI is based on the KITTI Vision Benchmark and distributed under CC BY-NC-SA 4.0. KITTI: "published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License" | https://www.semantic-kitti.org/dataset.html |
| Waymo Open Dataset v2 (3D semantic segmentation) | Waymo top-lidar scans; point classes and instances, instance velocities | Waymo Dataset License Agreement for Non-Commercial Use (March 2025): use for "research, teaching, scientific publication and personal experimentation"; "You may publish, in whole or in part, trained model architectures, including weights and biases" with the notice, a copy of the Agreement and attribution (Section 2(c)) | https://waymo.com/open/terms |
| PandaSet | Hesai Pandar64 scans, 10 Hz; point classes, instances from cuboids | "Unless specifically labeled otherwise, these Datasets are provided to You under a Creative Commons Attribution 4.0 International Public License ("CC BY 4.0"), with the additional terms included in these Dataset Terms." (PandaSet Dataset Terms of Use) | https://pandaset.org |

Argoverse 2, ONCE and Lyft Level 5 were only used to evaluate, not to train. Validation splits were never trained on.

## Multi-sweep input

The `sweeps` input holds the previous 4 scans of the same lidar, newest first, concatenated and padded to 262144 rows:
x, y, z in the **current** lidar frame, the raw intensity, and the time lag in seconds (current scan time minus that
scan's time, positive). The caller moves each previous scan into the current lidar frame with its pose:
`p_current = inv(T_current) @ T_previous @ p_previous`, with the lidar poses `T` in a fixed frame (odometry or map).
Trained lags are about 0.1-0.41 s (10 Hz); leave out scans older than 0.45 s. With no history (the first scans after a
start), fill the input with padding: segmentation stays almost the same, but the velocity is not usable.

In this repository `example_sweep_1.pcd` .. `example_sweep_4.pcd` are the 4 scans before `example.pcd`, moved into its
lidar frame with the car's recorded GNSS/INS localization, with the 2.7 m minimum range applied in their own frame.
`sweeps.json` (`{"lag": [0.1, 0.2, 0.3, 0.4]}`) gives their lags, rounded from the recorded 0.0995, 0.1997, 0.2999 and
0.3995 s. All are binary PCD with float32 x y z intensity, the intensity raw as from the Velodyne VLP-32C (10 Hz,
mounted 2.11 m above the ground on the car's roof).

## Results

**Public validation splits.** Each dataset evaluated with its own style embedding and class head, 4 previous sweeps,
300 evenly spaced frames per split, PyTorch fp32 at the final epoch. Obstacles are labelled points up to 2 m above the
ground within 2.7-75 m.

| Dataset | Validation frames (used / split) | pseg_v6.1: PQ / obstacle recall 50-75 m | pseg_v2 (previous, single scan): PQ / recall 50-75 m |
|---|---|---|---|
| nuScenes (Panoptic nuScenes val) | 300 / 6019 | 0.688 / 0.971 | 0.680 / 0.958 |
| SemanticKITTI (sequence 08) | 300 / 4071 | 0.826 / 0.984 | 0.822 / 0.973 |
| Waymo Open Dataset v2 (segmentation val) | 300 / 5976 | 0.845 / 0.977 | 0.840 / 0.977 |
| PandaSet (held-out sequences) | 300 / 1200 | 0.804 / 0.989 | 0.797 / 0.988 |
| **mean** | | **0.791 / 0.980** | **0.785 / 0.974** |

- PQ: class-agnostic panoptic quality of the labelled object instances (matched at IoU > 0.5).
- Obstacle recall 50-75 m: share of labelled obstacle points at 50-75 m that the model keeps as obstacles. Ground
  removal and clustering, the pipeline the model replaces, reach 0.016 / 0.462 / 0.498 / 0.570 on the four sets.
- pseg_v2: the previous deployed model, the same network without previous scans and velocity.
- Learned velocity, mean error of matched objects (moving / static, m/s): PandaSet 0.85 / 0.24, Waymo 0.76 / 0.19,
  nuScenes 0.85 / 0.23.

**Evaluation-only datasets** (box-labelled, not trained on; unknown embedding, nuScenes head, 300 evenly spaced
validation frames, PyTorch fp32):

| Dataset | Validation frames (used / split) | Previous scans | pseg_v6.1: PQ / recall of box points | pseg_v2: PQ / recall of box points |
|---|---|---|---|---|
| Argoverse 2 | 300 / 4806 | 4 (0.1-0.4 s) | 0.838 / 0.998 | 0.818 / 0.998 |
| ONCE | 300 / 3321 | none (the only cached one is 0.5 s old, outside the trained range) | 0.833 / 0.998 | 0.848 / 0.999 |
| Lyft Level 5 | 300 / 1260 | 2 (0.2 and 0.4 s) | 0.893 / 0.993 | 0.898 / 0.994 |

These sets have only 3D boxes: an object instance is the set of points inside a labelled box, and the rest of the scan
has no ground or static labels. PQ therefore depends on how tightly the boxes fit and which classes were labelled, and
obstacle recall is given for the box points only (the unlabelled rest includes the road).

**Our test set.** 300 labelled VLP-32C scans from Tartu on the lab's car, never trained on (box-labelled vehicles and
pedestrians within 50 m), through the deployed node code with TensorRT fp16:

| Model / setting | PQ | objects matched / missed | velocity error, moving / static objects (m/s) |
|---|---|---|---|
| ground removal + clustering | 0.569 | | |
| pseg_v2 (previous, deployed until October 2026) | 0.843 | 97.0% / 1.07% | |
| pseg_v6.1, TensorRT fp16 (deployed) | 0.857 | 97.2% / 0.58% | 1.03 / 0.33 |
| pseg_v6.1 without previous scans | 0.852 | | not usable (3.5 moving) |
| `pseg_v6.1_fp16.onnx`, ONNX Runtime CUDA (fp32 model: 0.861) | 0.857 | 97.2% / 0.58% | 1.04 / 0.33 |

## Deployment in autoware_mini

The model runs on the lab's car in [autoware_mini](https://github.com/UT-ADL/autoware_mini), the lab's ROS 1 autonomy
stack, as the `lidar_panoptic` detector: a ROS 1 node with ONNX Runtime and the TensorRT execution provider (fp16) at
the lidar's 10 Hz, about 27 ms per scan on an RTX 2080 Ti. It keeps the previous scans in the map frame, feeds the
model, and publishes static points to the collision checker and dynamic points with instance, class and velocity to
the clustering and tracking nodes. The node is under review and not yet in the public repository.

The sibling demo [lidar_pillar_detector](https://github.com/UT-ADL/lidar_pillar_detector) does the same for the lab's
3D box detector.

## License

- **Code** (`segment.py`): MIT, see [LICENSE](LICENSE).
- **Model weights** (`pseg_v6.1_fp16.onnx`, `pseg_v6.1_fp16.yaml`): licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), copyright 2026 University of Tartu,
  Autonomous Driving Lab, together with the Waymo Dataset License Agreement for Non-Commercial Use below. The weights
  were derived by training on nuScenes, SemanticKITTI, the Waymo Open Dataset and PandaSet (see
  [Styles and datasets](#styles-and-datasets)); the most restrictive of their terms apply: non-commercial use only,
  share-alike, attribution. Where the two licences differ, the more restrictive term applies.
- **Waymo Open Dataset notice** (required by Section 2(c) of its licence): This model was made using the Waymo Open
  Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at
  [waymo.com/open/terms](https://waymo.com/open/terms), and your access and use of such work are governed by the terms
  and conditions therein. This model is released under that Agreement, and any further downstream use or modification
  of it is subject to the terms of that Agreement, including the non-commercial restrictions of its Section 4.
  A copy of the Agreement is included as [WAYMO_LICENSE.txt](WAYMO_LICENSE.txt). Section 4(a) of the Agreement reads:
  "You must not use or deploy the Dataset, any models trained on or refined using the Dataset, or the weights or
  biases from such trained models, in whole or in part, (i) in operation of a vehicle or to assist in the operation of
  a vehicle, (ii) in any Production Systems, or (iii) for any other primarily commercial purposes".
- **Attribution** when you use or share the weights:
  - University of Tartu, Autonomous Driving Lab (the model);
  - nuScenes, nuScenes-lidarseg and Panoptic nuScenes (Motional): H. Caesar et al., "nuScenes: A multimodal dataset
    for autonomous driving", CVPR 2020; W. K. Fong et al., "Panoptic nuScenes: A Large-Scale Benchmark for LiDAR
    Panoptic Segmentation and Tracking", RA-L 2022;
  - SemanticKITTI: J. Behley et al., "SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences",
    ICCV 2019; KITTI: A. Geiger, P. Lenz, R. Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark
    Suite", CVPR 2012;
  - the Waymo Open Dataset, provided by Waymo LLC: P. Sun et al., "Scalability in Perception for Autonomous Driving:
    Waymo Open Dataset", CVPR 2020;
  - PandaSet (Scale AI and Hesai): P. Xiao et al., "PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving",
    ITSC 2021.

  No dataset provider endorses this model or its use.
- **Example scans** (`example.pcd`, `example_sweep_*.pcd`, `sweeps.json`, `example.png`): the lab's own data,
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), credit: University of Tartu, Autonomous
  Driving Lab.

**Credits:** Autonomous Driving Lab, Institute of Computer Science, University of Tartu.
