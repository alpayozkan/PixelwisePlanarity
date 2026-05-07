# ZeroPlane Evaluation Datasets (NYU-v2 & 7-Scenes)

Reference for the bundled NYU-v2 and 7-Scenes plane evaluation sets shipped by the ZeroPlane authors. Each sample is a self-contained `.npz` — RGB, depth, plane parameters, and segmentation are all inside; no external dataset download is required.

Source: `huggingface.co/datasets/jcleo0428/ZeroPlane_dataset/` (`nyuv2_plane.zip`, `sevenscenes_plane.zip`)
Local root: `/cluster/home/aoezkan/planeseg/ZeroPlane/datasets/`

---

## At a glance

| Property | NYU-v2 | 7-Scenes |
|---|---|---|
| Directory | `nyuv2_plane/` | `sevenscenes_plane/` |
| Manifest | `nyuv2_plane_len654_test.json` | `sevenscenes_plane_len758_val.json` |
| File count | 654 | 758 |
| Total size | 3.4 GB | 4.6 GB |
| Per-sample size | ~5.3 MB | ~6.9 MB |
| File naming | `<id>_d2.npz` (id 0–653) | `val_<id>_d2.npz` (id 0–757) |
| Split label | test | val |
| Image format | `CV2_BGR` | `CV2_BGR` |
| Model resolution | 256×192 | 256×192 |
| Native resolution | 640×480 | 640×480 |

Both manifests share the same JSON schema, both versioned `2.0`. NYU manifest created `2024-11-12`, 7-Scenes `2024-11-13`.

---

## Per-sample npz contents

Each `.npz` is a zip of `.npy` arrays. The two datasets share the same set of array names, but a few dtypes differ.

### Common arrays (both datasets)

| Array | NYU dtype | 7-Scenes dtype | Shape | Description |
|---|---|---|---|---|
| `image` | uint8 | uint8 | (192, 256, 3) | RGB at model resolution (BGR channel order) |
| `raw_image` | uint8 | uint8 | (480, 640, 3) | RGB at native sensor resolution |
| `depth` | float64 | float64 | (1, 192, 256) | Depth in meters at model resolution |
| `raw_depth` | **float32** | **float64** | (192, 256) | Raw sensor depth at model resolution |
| `high_res_depth` | float64 | float64 | (1, 480, 640) | Native-resolution depth |
| `high_res_raw_depth` | **float32** | **float64** | (480, 640) | Native-resolution raw depth |
| `segmentation` | **int32** | **int64** | (192, 256) | Plane instance IDs (0 = non-planar) |
| `plane` | float64 | float64 | (N, 3) | Plane parameters, one row per instance (N = `num_planes`) |
| `num_planes` | int64 shape `(1,)` | int64 scalar `()` | — | Number of planes |
| `intrinsic` | **float64** | **int64** | (3, 3) | Camera K for the model-resolution frame |
| `origin_img_path` | `<U100` | `<U73` | scalar | Provenance pointer (NOT a usable path) |

The 7-Scenes pack uses heavier dtypes (float64 vs float32 for raw depth, int64 vs int32 for segmentation), which is why its files are ~30% larger despite having the same image dimensions.

### Plane parameters (`plane` array)

Each row is a 3-vector encoding a plane. Convention follows ZeroPlane: `n / d` form where the row is the plane's outward normal scaled by `1/offset` (i.e., a 3D point `p` is on the plane iff `plane_row · p == 1`).

### Segmentation map

Integer instance IDs. `0` = non-planar / background. Positive IDs index into the `plane` array (segmentation id `k+1` corresponds to `plane[k]`). Use `cv2.INTER_NEAREST` if resizing.

### Provenance

`origin_img_path` is the location of the source image on the dataset authors' filesystem (e.g. `/data/Jiachen/3D_datasets/sevenscenes/chess/seq-03/frame-000498.color.png`). It's metadata only — the code never reads from this path. The actual RGB lives inside `raw_image` / `image`.

---

## Camera intrinsics

Both datasets ship a single shared K used for every sample (the manifest's top-level `camera` field). All values are in pixels for the **model-resolution 256×192 frame**.

### NYU-v2

```
fx = 207.54316047    fy = 207.78784445
cx = 130.23297976    cy = 101.49446653
```

### 7-Scenes (integer-valued)

```
fx = 234    fy = 234
cx = 128    cy =  96
```

Each per-sample npz also stores its own `intrinsic` array; in practice this matches the manifest K.

For native-resolution arrays (`raw_image`, `high_res_depth`, …) scale K by 640/256 = 2.5 in x and 480/192 = 2.5 in y.

---

## Manifest JSON schema

Both manifests have identical top-level structure:

```json
{
  "info":       { "description": "...", "version": "2.0", "data_created": "..." },
  "camera":     [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "categories": [{ "id": 1, "name": "plane" }],
  "annotations": [ {...}, ... ]
}
```

Each entry in `annotations[]`:

| Field | Type | Description |
|---|---|---|
| `image_id` | str | e.g. `"0_d2"`, `"123_d2"` |
| `origin_img_path` | str | Provenance only |
| `image_format` | str | `"CV2_BGR"` |
| `width`, `height` | int | 256, 192 |
| `npz_file_name` | str | Relative path to the npz |
| `segments_info` | list | Per-plane metadata (see below) |

`segments_info[i]`:

| Field | Description |
|---|---|
| `id` | Plane index (0-based, within this image) |
| `bbox` | `[x, y, w, h]` |
| `bbox_mode` | `1` = `BoxMode.XYWH_ABS` |
| `iscrowd` | `0` |
| `area` | Pixel area |
| `center` | Normalized centroid `[cx/W, cy/H]` |

---

## Dataset composition

### NYU-v2

- Single pooled test split. 654 images.
- Plane count per image: min=1, max=14, mean=7.4, median=7.

### 7-Scenes

- Pooled validation split across 6 scenes (the canonical 7-Scenes benchmark has 7; `stairs` is absent here).
- 758 images, plane count per image: min=1, max=20, mean=7.2, median=7.

| Scene | Frames |
|---|---|
| office | 157 |
| chess | 152 |
| fire | 141 |
| pumpkin | 109 |
| heads | 105 |
| redkitchen | 94 |
| **total** | **758** |

---

## How ZeroPlane consumes the data

Dataset mappers (Detectron2-style):
- `ZeroPlane/ZeroPlane/data/dataset_mappers/nyuv2_plane_dataset_mapper.py`
- `ZeroPlane/ZeroPlane/data/dataset_mappers/sevenscenes_plane_dataset_mapper.py`

Both follow the same load pattern (`nyuv2_plane_dataset_mapper.py:145`):

```python
data = np.load(dataset_dict["npz_file_name"])
image = data['raw_image'] if use_raw else data['image']    # uint8 BGR
seg   = data['segmentation']                                # int instance map
plane = data['plane']                                       # (N, 3) plane params
depth = data['depth']                                       # meters
K     = data['intrinsic']                                   # 3×3
```

Pre-built eval entry points:
- `ZeroPlane/scripts/eval_dust3r_nyu.sh`
- `ZeroPlane/scripts/eval_dust3r_sevenscenes.sh`

These set `DETECTRON2_DATASETS` to `ZeroPlane/datasets/` so the registered datasets resolve to the local paths above.

---

## Quick verification snippet

```python
import numpy as np
d = np.load("/cluster/home/aoezkan/planeseg/ZeroPlane/datasets/nyuv2_plane/0_d2.npz")
print(list(d.keys()))                   # 11 arrays
print(d['raw_image'].shape, d['raw_image'].dtype)   # (480, 640, 3) uint8
print(d['plane'].shape)                 # (N, 3)
print(int(d['num_planes']))             # N
print(d['intrinsic'])                   # 3×3 K
```

Same command works for `sevenscenes_plane/val_0_d2.npz`.
