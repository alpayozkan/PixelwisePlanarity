# Hypersim Dataset Verification

This directory contains tools for verifying the Hypersim dataset setup.

## Quick Start

Run the comprehensive verification script:

```bash
cd /cluster/home/aoezkan/planeseg/PixelwisePlanarity
bash planamono/shared/datasets/verify_hypersim_dataset.sh
```

This will:
- ✅ Check all data paths exist
- ✅ Validate split files (train/val/test)
- ✅ Verify sample scene structure
- ✅ Check HDF5 file formats
- ✅ Test dataset loading
- ✅ Verify no split overlap
- 📊 Provide comprehensive summary report

## Expected Output

### Success Case
```
================================================================================
VERIFICATION SUMMARY
================================================================================

Total checks performed: 15
Passed: 15
Failed: 0
Warnings: 0

✓ ALL CHECKS PASSED!

Your Hypersim dataset is ready to use!
```

### Failure Case
```
✗ SOME CHECKS FAILED

Please review the errors above and:
  1. Verify your data paths are correct
  2. Check that files follow the expected naming convention
  3. See docs/hypersim_dataset_setup.md for troubleshooting
```

## Individual Tests

### Test 1: Quick Python Test
```bash
python planamono/shared/datasets/test_hypersim_dataset.py
```

### Test 2: Split Verification Only
```bash
python planamono/shared/datasets/verify_hypersim_splits.py
```

### Test 3: Manual HDF5 Check
```bash
python -c "
import h5py

# Check structure
with h5py.File('/path/to/scene/cam_00_merged.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('RGB shape:', f['rgb'].shape)
    print('Depth shape:', f['depth'].shape)
"
```

## Files in This Directory

| File | Purpose |
|------|---------|
| `hypersim_plane_dataset.py` | Main dataset class |
| `verify_hypersim_dataset.sh` | Comprehensive verification script (bash) |
| `verify_hypersim_splits.py` | Split verification script (Python) |
| `test_hypersim_dataset.py` | Quick dataset test (Python) |
| `README_HYPERSIM.md` | This file |

## Troubleshooting

See [docs/hypersim_dataset_setup.md](../../../docs/hypersim_dataset_setup.md) for detailed troubleshooting guide.

Common issues:
- **Missing files**: Check naming conventions match expected patterns
- **Wrong HDF5 keys**: Verify your files have 'rgb', 'depth', 'planes', 'frame_ids'
- **Path errors**: Update paths in verification script if your setup differs
