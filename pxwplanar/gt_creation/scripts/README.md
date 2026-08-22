# GT Creation Scripts

Batch shell scripts for ground-truth generation. Scene lists are text files with
one scene id per line (`#` comments allowed). All scripts carry commented
`#SBATCH` directives — uncomment, `mkdir -p logs`, submit with `sbatch <script> [args]`
(typical: `--time=8:00:00 --cpus-per-task=4 --mem-per-cpu=8G --gpus=1`).

## ScanNet++

### 1. Plane extraction (mesh → labeled plane mesh)

```bash
./scannetpp_plane_extraction.sh <scene_list> [config] [input_root] [output_root]
# config default: ../configs/scannetpp_default.yml
```

Region growing + EM + IRLS on the semantic mesh. Outputs per scene:
`planes.json` (plane parameters) and `planes.ply` (mesh with plane_id per face).

### 2. Plane rendering (mesh → rendered.h5)

```bash
./scannetpp_render_planes.sh <scene_list> [input_root] [plane_root] [output_root] [frame_skip]
# input_root  default: <scannetppv2_path>/data
# plane_root  default: paths.scannetpp_plane_path
# output_root default: paths.scannetpp_rend_plane_path
# frame_skip  default: 25
```

Calls `../scannetpp/render_scene.py`: raycasts `planes.ply` into every Nth iPhone
frame and writes `<scene>/rendered.h5` (`planes (N,H,W) uint16`, 0 = non-planar;
`frame_ids`) — the GT consumed by training and evaluation.

### 3. GT depth rendering (mesh → rendered_depth.h5)

```bash
./scannetpp_render_depth.sh <scene_list> [input_root] [output_root] [frame_skip]
# input_root  default: <scannetppv2_path>/data
# output_root default: paths.scannetpp_rend_plane_path
# frame_skip  default: 25 (keep equal to step 2 so the H5s stay index-aligned)
```

Calls `../scannetpp/render_depth.py`: raycasts Z-depth from the full scene mesh
(`mesh_aligned_0.05.ply`) into the same frames as step 2 — with the same camera
model as the label raycast, so the two H5s stay pixel-aligned — and writes
`<scene>/rendered_depth.h5` (`depth (N,H,W) uint16` millimeters; `frame_ids`) —
the GT depth the 3D metrics in `evaluate_all_baselines.py` backproject.
CPU-only, no GL required.

### 4. Video generation

```bash
./scannetpp_video_gen.sh <scene_list> [h5_root] [rgb_root] [output_root] [fps]
# fps default: 5
```

Visualization videos from the rendered plane H5s.

## Hypersim

### 1. Plane extraction

```bash
./hypersim_plane_extraction.sh <scene_list> [config] [input_root] [output_root]
# config default: ../configs/hypersim_default.yml
```

### 2. Plane rendering

```bash
./hypersim_render_planes.sh <scene_list> [params_root] [plane_root] [output_root] \
                            [frame_skip] [python_script] [metadata_csv]
# python_script default: ../hypersim/rendering.py (repo-relative)
# metadata_csv  default: pxwplanar/shared/datasets/metadata_camera_parameters.csv
```

Outputs per scene: `rendered_planes_<cam>.h5` (one per camera).

### 3. Raycasted depth

```bash
./hypersim_raycast_depth.sh <scene_list> [params_root] [plane_root] [output_root] \
                            [frame_skip] [python_script] [metadata_csv] [depth_type]
# depth_type: zdepth (default, -> *_raycast/) or euclidean (-> *_raycast_euc/)
```

## SYNTHIA / VKITTI2 (outdoor)

Plane extraction from depth + semantic segmentation; scene lists and roots come
from the dataset config (`output_root` is read from the YAML):

```bash
./synthia_plane_extraction.sh [--config ../configs/synthia_default.yml]
./vkitti2_plane_extraction.sh [--config ../configs/vkitti2_default.yml]
```
