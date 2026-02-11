# Dataset Documentation

This document provides detailed specifications for the ScanNet++ and Hypersim plane segmentation datasets.

---

## Table of Contents

1. [ScanNetPPPlaneDataset](#scannetppplanedataset)
2. [HypersimPlaneDataset](#hypersimplanedataset)
3. [Quick Comparison](#quick-comparison)
4. [Usage Examples](#usage-examples)

---

## ScanNetPPPlaneDataset

PyTorch Dataset class for loading ScanNet++ scenes with plane segmentation labels, depth maps, semantic labels, and camera parameters.

### Overview

**Location:** `planamono/shared/datasets/scannetpp.py`

**Purpose:** Evaluation and training on real-world indoor scenes from ScanNet++ dataset with ground truth plane segmentation from mesh raycasting.

**Key Features:**
- Real-world iPhone RGB images (~1296×968 native resolution)
- Ground truth plane labels from mesh raycasting
- Per-frame camera intrinsics and poses from COLMAP
- Semantic segmentation labels
- Rendered depth maps from meshes

### Constructor Parameters

```python
ScanNetPPPlaneDataset(
    rgb_root,              # str: Base directory for RGB images
    plane_label_root,      # str: Base directory for plane label HDF5 files
    sem_label_root,        # str: Base directory for semantic label HDF5 files
    depth_label_root,      # str: Base directory for depth HDF5 files
    split_txt_dir,         # str: Directory containing split files
    split='train',         # str: One of ['train', 'val', 'test']
    image_height=512,      # int: Target image height for resizing
    image_width=768,       # int: Target image width for resizing
    max_scenes=None        # int or None: Limit number of scenes (None = all)
)
```

### Directory Structure

Expected file organization:

```
rgb_root/
    <scene_id>/
        iphone/
            rgb/
                <frame_id>.jpg           # RGB images (e.g., frame_000123.jpg)
            pose_intrinsic_imu.json      # Camera parameters per frame

plane_label_root/
    <scene_id>/
        rendered.h5                      # Plane segmentation labels
                                        # Datasets: "planes", "frame_ids"

sem_label_root/
    <scene_id>/
        rendered_sem.h5                  # Semantic labels
                                        # Datasets: "sem", "frame_ids"

depth_label_root/
    <scene_id>/
        rendered_depth.h5                # Depth maps in millimeters
                                        # Datasets: "depth", "frame_ids"

split_txt_dir/
    nvs_sem_train_with_planes.txt       # Train scene IDs (one per line)
    nvs_sem_val_with_planes.txt         # Val scene IDs
    nvs_sem_test_with_planes.txt        # Test scene IDs
```

### HDF5 File Format

**Plane Labels HDF5** (`rendered.h5`):
```python
{
    "planes": (N, H, W) int32          # Plane instance labels (0 = non-planar)
    "frame_ids": (N,) bytes            # Frame identifiers (e.g., b"frame_000123")
}
```

**Semantic Labels HDF5** (`rendered_sem.h5`):
```python
{
    "sem": (N, H, W) int64             # Semantic class labels
    "frame_ids": (N,) bytes            # Frame identifiers
}
```

**Depth HDF5** (`rendered_depth.h5`):
```python
{
    "depth": (N, H, W) uint16          # Depth in millimeters
    "frame_ids": (N,) bytes            # Frame identifiers
}
```

### Camera Parameters JSON

**Format** (`pose_intrinsic_imu.json`):
```json
{
    "frame_000123": {
        "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],    // 3×3 matrix
        "aligned_pose": [[r11, r12, r13, tx], ...]              // 4×4 c2w matrix
    },
    ...
}
```

### Data Loading

The dataset internally:
1. Loads scene IDs from split file (`nvs_sem_{split}_with_planes.txt`)
2. Validates that all required files exist for each scene
3. Reads `frame_ids` from plane HDF5 to get valid frames
4. Matches RGB files with HDF5 indices
5. Loads camera parameters from JSON

### Return Format

Each `__getitem__(idx)` call returns a dictionary:

```python
{
    "image": torch.FloatTensor,      # Shape: (3, H, W), range [0, 1]
    "depth": torch.FloatTensor,      # Shape: (1, H, W), meters (converted from mm)
    "plane": torch.IntTensor,        # Shape: (1, H, W), plane IDs (0 = non-planar)
    "sem": torch.LongTensor,         # Shape: (1, H, W), semantic class IDs
    "rgb_path": str,                 # Absolute path to RGB file
    "K": torch.FloatTensor,          # Shape: (3, 3), camera intrinsics
    "c2w": torch.FloatTensor,        # Shape: (4, 4), camera-to-world pose
    "scene_id": str,                 # Scene identifier (e.g., "0a5c013435")
    "frame_idx": str,                # Frame identifier (e.g., "frame_000123")
}
```

### Data Processing Pipeline

1. **Plane Labels**: Loaded from HDF5, negative labels set to 0
2. **Semantics**: Loaded from HDF5 as int64
3. **Depth**: Loaded from HDF5, converted from mm to meters (`/ 1000.0`)
4. **RGB**: Loaded via OpenCV, converted BGR→RGB, resized to match label dimensions, normalized to [0, 1]
5. **Camera**: Loaded from JSON per frame

### Memory Considerations

- HDF5 files are opened/closed per access (no persistent handles)
- Chunked reading: `f["planes"][frame_idx]` loads only one frame
- Valid pairs list stored in memory: `(rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w)`

### Statistics (Full Dataset)

- **Train**: ~704 scenes
- **Val**: ~150 scenes
- **Test**: ~152 scenes
- **Frames per scene**: Variable (typically 50-200)
- **Total frames**: ~100K+
- **Native resolution**: ~1296×968 (resized to 768×512 for training)

### Example Instantiation

```python
from planamono.shared.datasets import ScanNetPPPlaneDataset

dataset = ScanNetPPPlaneDataset(
    rgb_root="/path/to/scannetpp/data",
    plane_label_root="/path/to/rendered_planes",
    sem_label_root="/path/to/rendered_sem",
    depth_label_root="/path/to/rendered_depth",
    split_txt_dir="/path/to/planamono/splits/scannetpp",
    split="val",
    image_height=512,
    image_width=768,
    max_scenes=10  # Optional: limit for debugging
)

print(f"Dataset size: {len(dataset)} frames")
print(f"Scenes: {dataset.scene_ids}")

sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Plane labels shape: {sample['plane'].shape}")
print(f"Unique planes: {torch.unique(sample['plane'])}")
```

---

## HypersimPlaneDataset

PyTorch Dataset class for loading Hypersim synthetic scenes with plane segmentation labels from rendered camera viewpoints.

### Overview

**Location:** `planamono/shared/datasets/hypersim_plane_dataset.py`

**Purpose:** Training and evaluation on photorealistic synthetic indoor scenes with perfect ground truth from rendering.

**Key Features:**
- Photorealistic synthetic RGB images (768×1024 native resolution)
- HDR tone-mapped color with robust handling
- Multiple cameras per scene (cam_00, cam_01, cam_02)
- Ground truth plane labels from mesh raycasting
- Per-camera intrinsics (computed or from metadata)

### Constructor Parameters

```python
HypersimPlaneDataset(
    hypersim_root,         # str: Root directory of Hypersim dataset
    plane_label_root,      # str: Root directory for plane label HDF5 files
    params_root,           # str: Root directory for camera parameters
    split_txt_dir,         # str: Directory containing split files
    split='train',         # str: One of ['train', 'val', 'test']
    metadata_csv=None,     # str or None: Path to metadata_camera_parameters.csv
    image_height=768,      # int: Target image height
    image_width=1024,      # int: Target image width
    max_scenes=None        # int or None: Limit number of scenes
)
```

### Directory Structure

Expected file organization:

```
hypersim_root/  (Hypersim_merged)
    <scene_id>/                              # e.g., ai_001_001
        images/
            scene_cam_00_final_hdf5/
                frame.0000.color.hdf5        # RGB images (HDR)
                frame.0001.color.hdf5
                ...
            scene_cam_00_geometry_hdf5/
                frame.0000.depth_meters.hdf5 # Depth in meters
                frame.0000.semantic.hdf5     # Semantic labels
                ...
            scene_cam_01_final_hdf5/
                ...
            scene_cam_02_final_hdf5/
                ...

plane_label_root/  (Hypersim_rendered)
    <scene_id>/
        rendered_planes_cam_00.h5            # Plane labels for cam_00
        rendered_planes_cam_01.h5            # Plane labels for cam_01
        rendered_planes_cam_02.h5            # Plane labels for cam_02

params_root/  (Hypersim_params)
    <scene_id>/
        _detail/
            cam_00/
                camera_keyframe_positions.hdf5
                camera_keyframe_orientations.hdf5
            cam_01/
                ...

split_txt_dir/
    train.txt                                # Train scene IDs (e.g., ai_001_001)
    val.txt                                  # Val scene IDs
    test.txt                                 # Test scene IDs
```

### HDF5 File Format

**Plane Labels HDF5** (`rendered_planes_cam_XX.h5`):
```python
{
    "planes": (N, H, W) int32              # Plane instance labels (0 = non-planar)
    "frame_ids": (N,) bytes or int         # Frame identifiers (e.g., b"0000")
}
```

**RGB HDF5** (`frame.XXXX.color.hdf5`):
```python
{
    "dataset": (H, W, 3) float32 or uint16 # HDR or LDR RGB data
}
```
Note: Dataset name varies by file; accessed via `list(f.keys())[0]`

**Depth HDF5** (`frame.XXXX.depth_meters.hdf5`):
```python
{
    "dataset": (H, W) float32              # Depth in meters
}
```

### Camera Intrinsics

**Method 1: From metadata CSV** (if `metadata_csv` provided):
```python
# Computed from projection matrix in metadata_camera_parameters.csv
fx = M_proj[0, 0] * 0.5 * width
fy = -M_proj[1, 1] * 0.5 * height
cx = M_proj[0, 2] * 0.5 * width + 0.5 * width
cy = -M_proj[1, 2] * 0.5 * height + 0.5 * height
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

**Method 2: Default intrinsics** (if metadata unavailable):
```python
# Standard Hypersim intrinsics for 1024×768
fx = fy = 886.81
cx = 512.0  # image_width / 2
cy = 384.0  # image_height / 2
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

### HDR Tone Mapping

Hypersim RGB images are HDR (float32). The dataset applies robust tone mapping:

```python
def _tonemap_rgb_robust(hdr, gamma=2.2):
    # 1. Replace inf/nan with 0
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Clip negative values
    hdr = np.maximum(hdr, 0.0)

    # 3. Normalize by 99th percentile (avoid outliers)
    max_val = np.percentile(hdr[hdr > 0], 99)
    hdr = hdr / max(max_val, 1e-6)
    hdr = np.clip(hdr, 0.0, 1.0)

    # 4. Apply gamma correction
    hdr = hdr ** (1.0 / gamma)

    return hdr  # Range: [0, 1]
```

### Return Format

Each `__getitem__(idx)` call returns a dictionary:

```python
{
    "image": torch.FloatTensor,      # Shape: (3, H, W), range [0, 1] (tone-mapped)
    "depth": torch.FloatTensor,      # Shape: (1, H, W), meters
    "plane": torch.IntTensor,        # Shape: (1, H, W), plane IDs (0 = non-planar)
    "sem": torch.LongTensor,         # Shape: (1, H, W), zeros (not used for Hypersim)
    "rgb_path": str,                 # Formatted string: "scene_id/cam_name/frame_id"
    "K": torch.FloatTensor,          # Shape: (3, 3), camera intrinsics
    "c2w": torch.FloatTensor,        # Shape: (4, 4), identity (poses not used)
    "scene_id": str,                 # Scene identifier (e.g., "ai_001_001")
    "frame_idx": str,                # Frame identifier (e.g., "0000")
}
```

**Important Note on `rgb_path`:**
Unlike ScanNetPP, `rgb_path` is a **formatted identifier string**, not an actual file path. Format: `"{scene_id}/{cam_name}/{frame_id}"` (e.g., `"ai_001_001/cam_00/0000"`). Actual paths must be reconstructed using `hypersim_root`.

### Data Loading

The dataset internally:
1. Loads scene IDs from split file (`{split}.txt`)
2. Discovers all plane HDF5 files (one per camera: `rendered_planes_cam_XX.h5`)
3. For each camera, reads `frame_ids` from plane HDF5
4. Validates that corresponding RGB and depth HDF5 files exist
5. Computes or loads camera intrinsics

### Data Processing Pipeline

1. **Plane Labels**: Loaded from HDF5, negative labels set to 0
2. **RGB**: Loaded from HDF5, dtype-aware processing:
   - `uint8`: Divide by 255
   - `uint16`: Divide by 65535
   - `float32/float64`: Apply HDR tone mapping
3. **Depth**: Loaded from HDF5, already in meters
4. **Semantics**: Not used (returns zeros)
5. **Camera Pose**: Not used (returns identity matrix)

### Memory Considerations

- HDF5 files opened/closed per access
- One plane HDF5 per camera (not per frame)
- Valid pairs stored: `(scene_id, cam_name, frame_idx, fid, rgb_path, depth_path, plane_h5, K)`

### Statistics (Full Dataset)

- **Train**: ~319 scenes
- **Val**: ~68 scenes
- **Test**: ~70 scenes
- **Cameras per scene**: Typically 3 (cam_00, cam_01, cam_02)
- **Frames per camera**: ~100 keyframes
- **Total frames**: ~137K+
- **Native resolution**: 1024×768

### Example Instantiation

```python
from planamono.shared.datasets import HypersimPlaneDataset

dataset = HypersimPlaneDataset(
    hypersim_root="/path/to/Hypersim_merged",
    plane_label_root="/path/to/Hypersim_rendered",
    params_root="/path/to/Hypersim_params",
    split_txt_dir="/path/to/planamono/splits/hypersim",
    split="val",
    metadata_csv="/path/to/metadata_camera_parameters.csv",  # Optional
    image_height=768,
    image_width=1024,
    max_scenes=5  # Optional: limit for debugging
)

print(f"Dataset size: {len(dataset)} frames")
print(f"Scenes: {dataset.scene_ids}")

sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"RGB path: {sample['rgb_path']}")  # Formatted string, not actual path
print(f"Scene/Camera/Frame: {sample['scene_id']}, {sample['frame_idx']}")
```

---

## Quick Comparison

| Feature | ScanNetPPPlaneDataset | HypersimPlaneDataset |
|---------|----------------------|---------------------|
| **Source** | Real iPhone scans | Synthetic rendering |
| **Resolution** | ~1296×968 → 768×512 | 1024×768 |
| **RGB Format** | JPEG files | HDF5 (HDR float32) |
| **Cameras/Scene** | 1 (iPhone) | 3 (cam_00/01/02) |
| **Depth** | Rendered from mesh (mm) | Perfect synthetic (m) |
| **Semantics** | Available | Not used |
| **Camera Poses** | COLMAP aligned poses | Identity (not used) |
| **Intrinsics** | Per-frame from JSON | Per-camera computed |
| **Split Files** | `nvs_sem_{split}_with_planes.txt` | `{split}.txt` |
| **RGB Path Return** | Absolute file path | Formatted identifier string |
| **Plane Labels** | `rendered.h5` | `rendered_planes_cam_XX.h5` |
| **Tone Mapping** | Not needed (JPEG) | Robust HDR tone mapping |

---

## Usage Examples

### Basic DataLoader Setup

```python
from torch.utils.data import DataLoader

# ScanNet++
scannetpp_dataset = ScanNetPPPlaneDataset(...)
scannetpp_loader = DataLoader(
    scannetpp_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Hypersim
hypersim_dataset = HypersimPlaneDataset(...)
hypersim_loader = DataLoader(
    hypersim_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### Training Loop Integration

```python
for batch in scannetpp_loader:
    images = batch["image"]        # (B, 3, H, W)
    planes = batch["plane"]        # (B, 1, H, W)
    depths = batch["depth"]        # (B, 1, H, W)
    K = batch["K"]                 # (B, 3, 3)

    # Your training code here
    planarity_pred = model(images)
    loss = criterion(planarity_pred, (planes > 0).float())
```

### Evaluation with Camera Parameters

```python
for batch in scannetpp_loader:
    images = batch["image"]
    planes_gt = batch["plane"]
    K = batch["K"]                 # Camera intrinsics
    c2w = batch["c2w"]             # Camera-to-world pose
    depth = batch["depth"]

    # Predict planarity
    planarity = model(images)

    # Backproject to 3D for evaluation
    from planamono.shared.plane_fitting import backproject_v2
    pts_world, labels, valid = backproject_v2(
        depth[0, 0].numpy(),
        K[0].numpy(),
        c2w[0].numpy(),
        planes_gt[0, 0].numpy()
    )
```

### Accessing Scene-Level Data

```python
# Get all frames from a specific scene
scene_id = "0a5c013435"
scene_frames = [
    i for i, pair in enumerate(dataset.valid_pairs)
    if pair[0].split("/")[-4] == scene_id  # ScanNet++
]

print(f"Scene {scene_id} has {len(scene_frames)} frames")

# Load first frame from scene
sample = dataset[scene_frames[0]]
```

### Debugging: Visualize Samples

```python
import matplotlib.pyplot as plt

sample = dataset[0]
image = sample["image"].permute(1, 2, 0).numpy()  # (H, W, 3)
plane = sample["plane"][0].numpy()                # (H, W)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title(f"RGB: {sample['scene_id']}/{sample['frame_idx']}")
axes[1].imshow(plane, cmap="tab20")
axes[1].set_title(f"Planes: {len(np.unique(plane)) - 1} instances")
plt.show()
```

---

## Notes and Best Practices

### Performance Tips

1. **Use `num_workers > 0`** in DataLoader for parallel loading
2. **HDF5 Thread Safety**: Set `export HDF5_USE_FILE_LOCKING=FALSE` if encountering locking issues
3. **Memory**: Each dataset keeps `valid_pairs` list in memory (~few MB)
4. **Batch Size**: ScanNet++ can handle larger batches due to JPEG compression

### Common Pitfalls

1. **Hypersim RGB Path**: Don't use `sample["rgb_path"]` as a file path directly - it's a formatted identifier. Reconstruct actual paths if needed.
2. **Depth Units**: ScanNet++ converts mm→m, Hypersim is already in meters
3. **Plane Label 0**: Always means non-planar background in both datasets
4. **Negative Labels**: Always clipped to 0 in `__getitem__`
5. **Missing Files**: Dataset skips scenes/frames with missing data and prints warnings

### Extending the Datasets

To add new modalities:
1. Add new HDF5 file paths to `valid_pairs` tuple
2. Load data in `__getitem__` following existing pattern
3. Add new key to return dictionary
4. Update docstrings and this documentation

---

## References

- **ScanNet++**: [scannetpp.github.io](https://scannetpp.github.io/)
- **Hypersim**: [github.com/apple/ml-hypersim](https://github.com/apple/ml-hypersim)
- **Ground Truth Generation**: See `docs/gt_generation.md`
- **Dataset Splits**: See `planamono/splits/README.md`
