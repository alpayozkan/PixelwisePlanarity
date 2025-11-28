# Dataset Scene Splits

This directory contains scene lists and train/validation/test splits for ScanNet++ and Hypersim datasets.

## Directory Structure

```
splits/
├── scannetpp/
│   ├── all_scenes.txt           # All 1006 ScanNet++ scenes
│   ├── train.txt                # Training split (704 scenes, 70%)
│   ├── val.txt                  # Validation split (150 scenes, 15%)
│   └── test.txt                 # Test split (152 scenes, 15%)
├── hypersim/
│   ├── all_scenes.txt           # All 457 Hypersim scenes
│   ├── train.txt                # Training split (319 scenes, 70%)
│   ├── val.txt                  # Validation split (68 scenes, 15%)
│   ├── test.txt                 # Test split (70 scenes, 15%)
│   ├── split_scenes.py          # Script to create batch processing splits
│   └── scene_splits/            # 100 splits for parallel processing
│       ├── split_0/
│       │   └── scene_list_0.txt
│       ├── split_1/
│       │   └── scene_list_1.txt
│       └── ...
└── create_train_val_test_splits.py  # Script to regenerate splits
```

## Dataset Statistics

### ScanNet++
- **Total scenes**: 1,006
- **Train**: 704 scenes (70.0%)
- **Validation**: 150 scenes (14.9%)
- **Test**: 152 scenes (15.1%)
- **Scene ID format**: `0a7cc12c0e`, `0a5c013435`, etc. (10-character hex)

### Hypersim
- **Total scenes**: 457
- **Train**: 319 scenes (69.8%)
- **Validation**: 68 scenes (14.9%)
- **Test**: 70 scenes (15.3%)
- **Scene ID format**: `ai_001_001`, `ai_001_002`, etc.

## Usage

### Loading Splits in Python

```python
# Load train scenes
with open('splits/scannetpp/train.txt', 'r') as f:
    train_scenes = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(train_scenes)} training scenes")

# Or using pathlib
from pathlib import Path

split_file = Path('splits/hypersim/val.txt')
val_scenes = split_file.read_text().strip().split('\n')
```

### Using with PyTorch Datasets

```python
from shared.datasets import ScanNetPPPlaneDataset, HypersimPlanarityDataset

# Load ScanNet++ training data
with open('splits/scannetpp/train.txt', 'r') as f:
    train_scenes = [line.strip() for line in f]

train_dataset = ScanNetPPPlaneDataset(
    rgb_root='/path/to/scannetpp/data',
    plane_label_root='/path/to/plane_labels',
    scene_list=train_scenes,  # Pass scene list directly
    split='train'
)

# Load Hypersim validation data
val_dataset = HypersimPlanarityDataset(
    data_root='/path/to/hypersim',
    split_file='splits/hypersim/val.txt',
    split='val'
)
```

### Batch Processing with Hypersim Splits

For parallel GT generation or processing, use the 100-way splits:

```bash
# Process split 0
python gt_creation/hypersim/scene_runner.py \
    --scene_list splits/hypersim/scene_splits/split_0/scene_list_0.txt \
    --config configs/hypersim_default.yml

# Submit all splits to SLURM
for i in {0..99}; do
    sbatch --export=SPLIT_ID=$i gt_creation/scripts/hypersim_plane_extraction.sh
done
```

## Regenerating Splits

To regenerate train/val/test splits with different ratios or seed:

```python
python create_train_val_test_splits.py
```

Edit the script to modify:
- `train_ratio`: Training set proportion (default: 0.7)
- `val_ratio`: Validation set proportion (default: 0.15)
- `test_ratio`: Test set proportion (default: 0.15)
- `seed`: Random seed for reproducibility (default: 42)

### Creating Custom Batch Splits

To create custom batch splits for parallel processing:

```python
cd hypersim
python split_scenes.py

# Modify split_scenes.py:
# - N: Number of splits (default: 100)
# - REVERSE_ORDER: Reverse scene order (default: False)
```

## Split Creation Details

### Train/Val/Test Splits

- **Random seed**: 42 (for reproducibility)
- **Shuffling**: Scenes are shuffled before splitting
- **Ratios**: 70% train, 15% val, 15% test
- **No overlap**: Each scene appears in exactly one split

### Batch Processing Splits (Hypersim)

- **Total splits**: 100
- **Scenes per split**: ~4-5 scenes each
- **Purpose**: Parallel GT generation on SLURM cluster
- **Distribution**: Scenes evenly distributed across splits

## File Formats

All split files use simple text format:
- One scene ID per line
- UTF-8 encoding
- Unix line endings (LF)
- No header, no empty lines at EOF

### Example (scannetpp/train.txt)
```
0a7cc12c0e
0a5c013435
0acbcdc1d0
...
```

### Example (hypersim/val.txt)
```
ai_001_001
ai_002_003
ai_003_007
...
```

## Validation

Verify splits are valid:

```python
# Check no overlap between splits
def validate_splits(train_file, val_file, test_file):
    with open(train_file) as f:
        train = set(line.strip() for line in f)
    with open(val_file) as f:
        val = set(line.strip() for line in f)
    with open(test_file) as f:
        test = set(line.strip() for line in f)

    assert len(train & val) == 0, "Train/val overlap!"
    assert len(train & test) == 0, "Train/test overlap!"
    assert len(val & test) == 0, "Val/test overlap!"

    total = len(train) + len(val) + len(test)
    print(f"✓ No overlap. Total: {total} scenes")

# Run validation
validate_splits('scannetpp/train.txt', 'scannetpp/val.txt', 'scannetpp/test.txt')
validate_splits('hypersim/train.txt', 'hypersim/val.txt', 'hypersim/test.txt')
```

## Notes

1. **Scene IDs are stable**: These splits can be used across different experiments for fair comparison

2. **Reproducibility**: Fixed random seed (42) ensures splits are reproducible

3. **Balanced distribution**: Scenes are shuffled before splitting to ensure representative distribution

4. **Batch splits independent**: The 100-way batch splits are independent of train/val/test splits (used only for parallel processing)

5. **Custom splits**: You can create custom splits by editing scene lists manually or using the provided scripts

## Common Use Cases

### Training a Model
```python
# Use train.txt for training
train_dataset = load_dataset(split='train')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Validation During Training
```python
# Use val.txt for validation
val_dataset = load_dataset(split='val')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Final Evaluation
```python
# Use test.txt for final evaluation
test_dataset = load_dataset(split='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

### Parallel GT Generation
```bash
# Use scene_splits for batch processing
for split_id in {0..99}; do
    process_scenes "hypersim/scene_splits/split_${split_id}/scene_list_${split_id}.txt"
done
```

## References

- ScanNet++ dataset: https://kaldir.vc.in.tum.de/scannetpp/
- Hypersim dataset: https://github.com/apple/ml-hypersim
