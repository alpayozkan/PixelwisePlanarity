# Shell Scripts Location Guide

All shell scripts have been created and organized in their respective `scripts/` directories.

## Location Overview

```
clean_structure/
├── gt_creation/scripts/          # 6 shell scripts + README
│   ├── scannetpp_plane_extraction.sh
│   ├── scannetpp_render_planes.sh
│   ├── hypersim_plane_extraction.sh
│   ├── hypersim_render_planes.sh
│   ├── batch_submit.sh
│   └── README.md
│
├── evaluation/scripts/            # 3 shell scripts + README
│   ├── run_evaluation.sh
│   ├── batch_evaluate.sh
│   └── README.md
│
└── inference/scripts/             # (Future: inference runners)
```

---

## GT Creation Scripts

**Location:** `gt_creation/scripts/`

### ScanNet++ Workflows

**1. Extract planes from meshes:**
```bash
cd gt_creation/scripts
./scannetpp_plane_extraction.sh scene_list.txt
```

**2. Render planes to images:**
```bash
./scannetpp_render_planes.sh scene_list.txt
```

### Hypersim Workflows

**1. Extract planes from meshes:**
```bash
./hypersim_plane_extraction.sh scene_list.txt
```

**2. Render planes to HDF5:**
```bash
./hypersim_render_planes.sh scene_list.txt
```

### Batch Processing (SLURM)

**Submit multiple splits to cluster:**
```bash
./batch_submit.sh scene_splits/ scannetpp_plane_extraction.sh scannetpp
```

**Full Documentation:** See `gt_creation/scripts/README.md`

---

## Evaluation Scripts

**Location:** `evaluation/scripts/`

### Single Method Evaluation

```bash
cd evaluation/scripts
./run_evaluation.sh moge test ./results/moge_test
```

**Methods:** `gt`, `moge`, `planercnn`, `monoplane`

### Batch Evaluation

**Evaluate all methods:**
```bash
./batch_evaluate.sh test ./results
```

**Full Documentation:** See `evaluation/scripts/README.md`

---

## Quick Start Examples

### Example 1: Generate GT for 5 ScanNet++ scenes

```bash
cd gt_creation/scripts

# Create scene list
cat > test_scenes.txt << EOF
0a5c013435
0a7cc12c0e
0ad96a1552
EOF

# Extract planes
./scannetpp_plane_extraction.sh test_scenes.txt \
    ../configs/scannetpp_default.yml \
    /path/to/scannetpp/data \
    /path/to/output

# Render to images
./scannetpp_render_planes.sh test_scenes.txt \
    /path/to/scannetpp/data \
    /path/to/output \
    /path/to/rendered
```

### Example 2: Evaluate MoGe on test set

```bash
cd evaluation/scripts

# Set model path
export MOGE_MODEL_PATH=/path/to/moge/checkpoint.pth

# Run evaluation
./run_evaluation.sh moge test ./results/moge_test
```

### Example 3: Batch processing on SLURM cluster

```bash
cd gt_creation/scripts

# Organize scenes into splits
mkdir -p scene_splits/split_{0,1,2}
split -l 50 all_scenes.txt scene_splits/split_
for i in 0 1 2; do
    mv scene_splits/split_a$i scene_splits/split_$i/scene_list_$i.txt
done

# Submit all splits
./batch_submit.sh scene_splits/ scannetpp_plane_extraction.sh scannetpp

# Monitor
squeue -u $USER
tail -f logs/split_0.out
```

---

## Script Features

### All scripts support:
-  **SLURM integration** - Comment/uncomment `#SBATCH` directives
-  **Local execution** - Work without SLURM
-  **Command-line args** - Configurable paths
-  **Error handling** - Validates inputs, reports failures
-  **Progress logging** - Clear status messages
-  **Batch processing** - Process multiple scenes
-  **Scene list format** - One scene ID per line, comments allowed

### Example scene list format:
```txt
# ScanNet++ test scenes
0a5c013435
0a7cc12c0e
# Skip problematic scenes
# 0ad96a1552
0b22fa63d2
```

---

## Script Dependencies

### GT Creation Scripts require:
- Python environment with dependencies
- `scene_runner.py` in `gt_creation/scannetpp/` or `hypersim/`
- Config YAML files in `gt_creation/configs/`
- Dataset paths (ScanNet++, Hypersim)

### Evaluation Scripts require:
- `run_evaluation.py` in `evaluation/`
- Model checkpoints (for MoGe, etc.)
- Ground truth data
- GPU (optional, for MoGe)

---

## Full Documentation

Each `scripts/` directory has a detailed `README.md`:

1. **`gt_creation/scripts/README.md`**
   - Pipeline order
   - SLURM configuration
   - Troubleshooting
   - Parameter tuning

2. **`evaluation/scripts/README.md`**
   - Metrics explained
   - Output format
   - Visualization
   - Troubleshooting

---

## Script Checklist

All scripts are:
-  **Executable** (`chmod +x`)
-  **Documented** (inline comments + README)
-  **Tested format** (valid bash syntax)
-  **SLURM-ready** (with commented directives)
-  **Standalone** (can run independently)

---

## Migration from Old Scripts

### Old structure:
```
gt_gen/run_processing.sh
gt_gen/run_raycast_plane.sh
gt_gen/submit_all_splits.sh
plane_fitting/eval_run.sh
```

### New structure:
```
gt_creation/scripts/scannetpp_plane_extraction.sh
gt_creation/scripts/scannetpp_render_planes.sh
gt_creation/scripts/batch_submit.sh
evaluation/scripts/run_evaluation.sh
```

