import random
from pathlib import Path

# Config
SPLIT_RATIO = 0.95  # 95% train, 5% val
SEED = 42

script_dir = Path(__file__).parent

# Read current train scenes (these have plane annotations)
train_file = script_dir / 'nvs_sem_train_with_planes.txt'
with open(train_file, 'r') as f:
    train_scenes = [line.strip() for line in f if line.strip()]

print(f"Total train scenes: {len(train_scenes)}")

# Shuffle with seed for reproducibility
random.seed(SEED)
random.shuffle(train_scenes)

# Split
split_idx = int(len(train_scenes) * SPLIT_RATIO)
new_train = sorted(train_scenes[:split_idx])
new_val = sorted(train_scenes[split_idx:])

print(f"New train: {len(new_train)} scenes ({SPLIT_RATIO*100:.0f}%)")
print(f"New val: {len(new_val)} scenes ({(1-SPLIT_RATIO)*100:.0f}%)")

# Write new splits
new_train_file = script_dir / 'nvs_sem_train_with_planes_fixed.txt'
new_val_file = script_dir / 'nvs_sem_val_with_planes_fixed.txt'

with open(new_train_file, 'w') as f:
    f.write('\n'.join(new_train) + '\n')
print(f"Written: {new_train_file}")

with open(new_val_file, 'w') as f:
    f.write('\n'.join(new_val) + '\n')
print(f"Written: {new_val_file}")

# Current val becomes test (untouched during training)
old_val_file = script_dir / 'nvs_sem_val_with_planes.txt'
test_file = script_dir / 'nvs_sem_test_with_planes.txt'

with open(old_val_file, 'r') as f:
    test_scenes = [line.strip() for line in f if line.strip()]

with open(test_file, 'w') as f:
    f.write('\n'.join(sorted(test_scenes)) + '\n')

print(f"Test (from old val): {len(test_scenes)} scenes")
print(f"Written: {test_file}")

print("\nSummary:")
print(f"  Train: {len(new_train)} scenes -> {new_train_file.name}")
print(f"  Val:   {len(new_val)} scenes -> {new_val_file.name}")
print(f"  Test:  {len(test_scenes)} scenes -> {test_file.name}")
