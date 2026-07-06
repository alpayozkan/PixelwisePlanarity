# GT Creation Scripts

Shell scripts for batch processing ground truth generation.

## ScanNet++ Scripts

### 1. Plane Extraction
```bash
./scannetpp_plane_extraction.sh scene_list.txt \
    ../configs/scannetpp_default.yml \
    /path/to/scannetpp/data \
    /path/to/output
```

Extracts planes from semantic meshes using region growing + EM algorithm.

**Inputs:**
- `scene_list.txt` - One scene ID per line
- Config YAML with extraction parameters
- ScanNet++ data root (contains scene folders)
- Output root for planes.json and planes.ply

**Outputs:**
- `<scene_id>/planes.json` - Plane parameters (normal, d, area, etc.)
- `<scene_id>/planes.ply` - Mesh with plane_id per face

### 2. Plane Rendering
```bash
./scannetpp_render_planes.sh scene_list.txt \
    /path/to/scannetpp/data \
    /path/to/plane/output \
    /path/to/rendered/output
```

Raycasts extracted planes to 2D images using camera poses.

**Outputs:**
- `<scene_id>/rendered_planes.h5` - HDF5 with per-frame plane labels

## Hypersim Scripts

### 1. Plane Extraction
```bash
./hypersim_plane_extraction.sh scene_list.txt \
    ../configs/hypersim_default.yml \
    /path/to/hypersim/dataset \
    /path/to/output
```

Extracts planes from Hypersim meshes.

**Outputs:**
- `<scene_id>/planes.json`
- `<scene_id>/planes.ply`

### 2. Plane Rendering
```bash
./hypersim_render_planes.sh scene_list.txt \
    /path/to/hypersim/dataset \
    /path/to/plane/output \
    /path/to/rendered/output
```

Renders planes to HDF5 for all cameras in each scene.

**Outputs:**
- `<scene_id>/rendered_planes_cam_00.h5`
- `<scene_id>/rendered_planes_cam_01.h5` (if exists)


## Scene List Format

One scene ID per line:
```
0a5c013435
0a7cc12c0e
0ad96a1552
# Comments allowed
```

## SLURM Configuration

Scripts include commented SLURM directives. Uncomment and adjust for your cluster:

```bash
#SBATCH --time=72:00:00          # Max runtime
#SBATCH --cpus-per-task=16       # CPU cores
#SBATCH --mem-per-cpu=32G        # Memory per CPU
```

## Local Execution (No SLURM)

Scripts work locally too - just remove/comment SBATCH lines:

```bash
# Process a few scenes locally
head -5 all_scenes.txt > small_test.txt
./scannetpp_plane_extraction.sh small_test.txt
```

## Pipeline Order

1. **Extract planes** from meshes
   - `scannetpp_plane_extraction.sh` or `hypersim_plane_extraction.sh`
2. **Render to images**
   - `scannetpp_render_planes.sh` or `hypersim_render_planes.sh`
3. **Evaluate** (use `evaluation/` scripts)

## Troubleshooting

**Out of memory:**
- Reduce `--cpus-per-task` and increase `--mem-per-cpu`
- Enable large scene splitting in config: `large_split_enable: 1`

**Failed scenes:**
- Check `logs/split_*.err` for errors
- Re-run failed scenes only by editing scene list

**Slow processing:**
- Increase `jobs: 16` in YAML config for parallelism
- Use `backend: "processes"` instead of "threads"
