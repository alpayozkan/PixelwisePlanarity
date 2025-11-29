import os
import shutil

# --- Config ---
full_scene_list_path = "all_scenes.txt"   # Path to input scene list
output_root = "scene_splits"                   # Output folder
N = 10                                         # Number of partitions
REVERSE_ORDER = False                           # 🔁 Set to True to reverse scene order
# REVERSE_ORDER = True

# --- Clean old splits ---
if os.path.exists(output_root):
    print(f"[INFO] Removing old output folder: {output_root}")
    shutil.rmtree(output_root)

# --- Read full scene list ---
with open(full_scene_list_path, "r") as f:
    all_scenes = [line.strip() for line in f if line.strip()]

if REVERSE_ORDER:
    all_scenes = list(reversed(all_scenes))
    print(f"[INFO] Reversed scene list order")

# --- Partition scenes ---
total = len(all_scenes)
scenes_per_split = (total + N - 1) // N  # ceiling division
print('total: ', total)
print('scenes_per_split: ', scenes_per_split)
# --- Write splits ---
os.makedirs(output_root, exist_ok=True)

for i in range(N):
    start = i * scenes_per_split
    end = min(start + scenes_per_split, total)
    split_scenes = all_scenes[start:end]

    split_dir = os.path.join(output_root, f"split_{i}")
    os.makedirs(split_dir, exist_ok=True)

    split_path = os.path.join(split_dir, f"scene_list_{i}.txt")
    with open(split_path, "w") as f:
        # f.write("\n".join(split_scenes))
        f.write("\n".join(split_scenes) + "\n")

    print(f"[INFO] Split {i}: {len(split_scenes)} scenes -> {split_path}")


    

# import os
# import shutil

# # --- Config ---
# full_scene_list_path = "scene_list_full.txt"  # Input file
# output_root = "scene_splits"                  # Where to write partitions
# N = 2                                         # Number of partitions

# # --- Clean old splits ---
# if os.path.exists(output_root):
#     print(f"[INFO] Removing old output folder: {output_root}")
#     shutil.rmtree(output_root)
    
# # --- Read full scene list ---
# with open(full_scene_list_path, "r") as f:
#     all_scenes = [line.strip() for line in f if line.strip()]

# total = len(all_scenes)
# scenes_per_split = (total + N - 1) // N  # ceiling division

# # --- Partition and write ---
# os.makedirs(output_root, exist_ok=True)

# for i in range(N):
#     start = i * scenes_per_split
#     end = min(start + scenes_per_split, total)
#     split_scenes = all_scenes[start:end]

#     split_dir = os.path.join(output_root, f"split_{i}")
#     os.makedirs(split_dir, exist_ok=True)

#     split_path = os.path.join(split_dir, f"scene_list_{i}.txt")
#     with open(split_path, "w") as f:
#         f.write("\n".join(split_scenes))

#     print(f"[INFO] Split {i}: {len(split_scenes)} scenes -> {split_path}")