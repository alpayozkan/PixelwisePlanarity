# Hypersim Dataset Setup Guide

This document explains how to use the new `HypersimPlaneDataset` class for evaluation, similar to `ScanNetPPPlaneDataset`.

## Overview

The `HypersimPlaneDataset` class has been created to enable evaluation on Hypersim data with the same pipeline used for ScanNet++.

**Location**: `planamono/shared/datasets/hypersim_plane_dataset.py`

## Expected Data Structure

The dataset expects the following directory structure:

```
Hypersim_merged/               # RGB and depth data
    <scene_id>/
        <cam_name>_merged.h5   # Contains 'rgb' and 'depth' datasets
                              # Example: cam_00_merged.h5, cam_01_merged.h5

Hypersim_rendered/            # Plane labels
    <scene_id>/
        rendered_planes_<cam_name>.h5  # Contains 'planes' and 'frame_ids'
                                       # Example: rendered_planes_cam_00.h5

Hypersim_params/              # Camera intrinsics
    <scene_id>/
        <cam_name>_intrinsics.json     # Contains 'K' matrix (3x3)
                                       # Example: cam_00_intrinsics.json
```

### HDF5 Dataset Keys

**RGB/Depth file (`<cam_name>_merged.h5`):**
- `rgb`: (N_frames, H, W, 3) - RGB images, dtype: uint8 [0-255] or float32 [0-1]
- `depth`: (N_frames, H, W) - Depth in meters, dtype: float32

**Plane labels file (`rendered_planes_<cam_name>.h5`):**
- `planes`: (N_frames, H, W) - Plane instance IDs, dtype: int32
- `frame_ids`: (N_frames,) - Frame ID strings, dtype: S (bytes) or str

**Intrinsics JSON (`<cam_name>_intrinsics.json`):**
```json
{
  "K": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ]
}
```

## Data Preparation

### Step 1: Verify Your Data Paths

The user-provided paths are:
```python
rgb_depth_root = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
plane_label_root = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
intrinsics_root = "/cluster/scratch/ayavuz/dataset/Hypersim_params"
split_txt_dir = "planamono/splits/hypersim"
```

**Action Required**: Verify that these directories exist and follow the expected structure.

### Step 2: Check File Naming Conventions

The dataset expects specific naming patterns. If your files are named differently, you'll need to either:

1. **Rename your files** to match the expected pattern, OR
2. **Modify the dataset class** to match your naming convention

**Expected patterns:**
- RGB/Depth: `{scene_id}/{cam_name}_merged.h5`
- Planes: `{scene_id}/rendered_planes_{cam_name}.h5`
- Intrinsics: `{scene_id}/{cam_name}_intrinsics.json`

### Step 3: Verify HDF5 Structure

Check that your HDF5 files have the correct keys:

```python
import h5py

# Check RGB/Depth file
with h5py.File("path/to/cam_00_merged.h5", "r") as f:
    print(f.keys())  # Should have 'rgb' and 'depth'
    print(f["rgb"].shape)    # (N_frames, H, W, 3)
    print(f["depth"].shape)  # (N_frames, H, W)

# Check plane file
with h5py.File("path/to/rendered_planes_cam_00.h5", "r") as f:
    print(f.keys())  # Should have 'planes' and 'frame_ids'
    print(f["planes"].shape)     # (N_frames, H, W)
    print(f["frame_ids"].shape)  # (N_frames,)
```

## Testing the Dataset

### Quick Test

Use the provided test script:

```bash
cd planamono/shared/datasets
python test_hypersim_dataset.py
```

This will:
1. Initialize the dataset with 2 validation scenes
2. Load a few samples
3. Test the DataLoader

**Expected output:**
```
================================================================================
Testing HypersimPlaneDataset
================================================================================
...
[Hypersim] val split → XX pairs from 2 scenes
✓ Dataset initialized successfully!
```

### Manual Test

```python
from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset

dataset = HypersimPlaneDataset(
    rgb_depth_root="/cluster/scratch/ayavuz/dataset/Hypersim_merged",
    plane_label_root="/cluster/scratch/ayavuz/dataset/Hypersim_rendered",
    intrinsics_root="/cluster/scratch/ayavuz/dataset/Hypersim_params",
    split_txt_dir="planamono/splits/hypersim",
    split='val',
    max_scenes=2  # Test with 2 scenes first
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample keys: {list(sample.keys())}")
print(f"Image shape: {sample['image'].shape}")
print(f"Plane shape: {sample['plane'].shape}")
```

## Running Evaluation

Once the dataset is working, use the evaluation script:

```bash
cd planamono/evaluation/quantitative
python evaluate_hypersim_fast.py
```

**Before running**, update the paths in `evaluate_hypersim_fast.py`:
- `model_path`: Path to your trained MoGe model
- `rgb_depth_root`: Path to Hypersim_merged
- `plane_label_root`: Path to Hypersim_rendered
- `intrinsics_root`: Path to Hypersim_params

## Troubleshooting

### Issue: "Missing RGB/depth for scene_id/cam_name"

**Cause**: The `{cam_name}_merged.h5` file doesn't exist.

**Solution**:
1. Check if the file exists at the expected path
2. Verify the camera name matches (e.g., `cam_00`, `cam_01`)
3. If your files are named differently, modify line 106 in `hypersim_plane_dataset.py`:
   ```python
   rgb_depth_h5_path = os.path.join(rgb_depth_scene_dir, f"{cam_name}_merged.h5")
   ```

### Issue: "Missing intrinsics for scene_id/cam_name"

**Cause**: The `{cam_name}_intrinsics.json` file doesn't exist.

**Solution**:
1. Check if intrinsics files exist
2. If you don't have per-camera intrinsics, you can create them from `metadata_camera_parameters.csv`:
   ```python
   # Create intrinsics JSON files from Hypersim metadata
   import pandas as pd
   import numpy as np
   import json

   df = pd.read_csv("metadata_camera_parameters.csv", index_col="scene_name")
   for scene_id in df.index:
       row = df.loc[scene_id]
       width = int(row["settings_output_img_width"])
       height = int(row["settings_output_img_height"])
       M_proj = np.array([[row[f"M_proj_{i}{j}"] for j in range(4)] for i in range(4)])

       # Convert to intrinsics
       fx = M_proj[0, 0] * 0.5 * width
       fy = -M_proj[1, 1] * 0.5 * height
       cx = M_proj[0, 2] * 0.5 * width + 0.5 * width
       cy = -M_proj[1, 2] * 0.5 * height + 0.5 * height
       K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

       # Save for each camera (all cameras have same intrinsics)
       scene_dir = f"Hypersim_params/{scene_id}"
       os.makedirs(scene_dir, exist_ok=True)
       for cam_name in ["cam_00", "cam_01", "cam_02"]:
           with open(f"{scene_dir}/{cam_name}_intrinsics.json", "w") as f:
               json.dump({"K": K}, f)
   ```

### Issue: "Error reading frame_ids from HDF5"

**Cause**: The `frame_ids` dataset in the plane HDF5 file has an unexpected format.

**Solution**: Check the dtype and format:
```python
with h5py.File("rendered_planes_cam_00.h5", "r") as f:
    print(f["frame_ids"].dtype)
    print(f["frame_ids"][0])  # Should be a string or bytes
```

If frame_ids are integers instead of strings, modify line 134 in `hypersim_plane_dataset.py`:
```python
frame_ids = [f"{fid:04d}" for fid in f["frame_ids"][:]]  # Format as 4-digit strings
```

### Issue: RGB images are too dark/bright

**Cause**: Hypersim uses HDR images that need tone mapping.

**Solution**: The dataset class assumes RGB is already tone-mapped in the merged file. If not, you'll need to:
1. Pre-process RGB with tone mapping before creating merged.h5, OR
2. Add tone mapping to the dataset class (see `HypersimPlanarityDataset.load_hypersim_rgb()` for example)

## Dataset Class Customization

If you need to modify the dataset class for your specific data structure, the key areas to change are:

1. **File paths** (lines 106-108):
   ```python
   rgb_depth_h5_path = os.path.join(rgb_depth_scene_dir, f"{cam_name}_merged.h5")
   intrinsics_json_path = os.path.join(intrinsics_scene_dir, f"{cam_name}_intrinsics.json")
   ```

2. **HDF5 keys** (lines 181, 195, 208):
   ```python
   plane = f["planes"][frame_idx]
   rgb = f["rgb"][frame_idx]
   depth = f["depth"][frame_idx]
   ```

3. **Frame ID format** (line 134):
   ```python
   frame_ids = [fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                for fid in f["frame_ids"][:]]
   ```

## Comparison with ScanNetPPPlaneDataset

| Aspect | ScanNetPPPlaneDataset | HypersimPlaneDataset |
|--------|----------------------|---------------------|
| RGB source | Individual JPG files | HDF5 dataset |
| Depth source | HDF5 per scene | HDF5 per camera |
| Plane labels | HDF5 per scene | HDF5 per camera |
| Intrinsics | JSON per scene | JSON per camera |
| Cameras | Single (iphone) | Multiple (cam_00, cam_01, ...) |
| Semantic labels | Separate HDF5 | Optional (zeros if not available) |
| Camera poses | Available | Set to identity (not used) |

## Next Steps

After successfully testing the dataset:

1. **Run full evaluation** on validation split
2. **Compare results** with ScanNet++ evaluation
3. **Add Hypersim evaluation** to `evaluate_all_baselines.py`
4. **Update documentation** with Hypersim-specific metrics

## Files Created

- `planamono/shared/datasets/hypersim_plane_dataset.py` - Dataset class
- `planamono/shared/datasets/test_hypersim_dataset.py` - Test script
- `planamono/evaluation/quantitative/evaluate_hypersim_fast.py` - Evaluation script
- `docs/hypersim_dataset_setup.md` - This document
