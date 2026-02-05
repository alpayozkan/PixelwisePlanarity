#!/bin/bash
################################################################################
# Hypersim Dataset Verification Script
#
# This script performs comprehensive checks on the Hypersim dataset structure:
# 1. Path existence verification
# 2. Sample scene structure check
# 3. HDF5 file format verification
# 4. Split file validation
# 5. Dataset loading test
# 6. Split overlap detection
#
# Usage: bash verify_hypersim_dataset.sh
################################################################################

# set -e  # Exit on error (disabled for debugging)

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
RGB_DEPTH_ROOT="/cluster/scratch/ayavuz/dataset/Hypersim_merged"
PLANE_LABEL_ROOT="/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
INTRINSICS_ROOT="/cluster/scratch/ayavuz/dataset/Hypersim_params"
SPLIT_TXT_DIR="planamono/splits/hypersim"
PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity"

# Counters for summary
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Helper functions
print_header() {
    echo -e "\n${BOLD}${BLUE}================================================================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}================================================================================${NC}"
}

print_subheader() {
    echo -e "\n${BOLD}$1${NC}"
}

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

################################################################################
# MAIN VERIFICATION STARTS HERE
################################################################################

cd "$PROJECT_ROOT" || exit 1

print_header "HYPERSIM DATASET VERIFICATION"

echo "Timestamp: $(date)"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Configuration:"
echo "  RGB/Depth root:     $RGB_DEPTH_ROOT"
echo "  Plane label root:   $PLANE_LABEL_ROOT"
echo "  Intrinsics root:    $INTRINSICS_ROOT"
echo "  Split txt dir:      $SPLIT_TXT_DIR"

################################################################################
# CHECK 1: Path Existence
################################################################################

print_header "CHECK 1: Path Existence"

if [ -d "$RGB_DEPTH_ROOT" ]; then
    check_pass "RGB/Depth root exists: $RGB_DEPTH_ROOT"
    RGB_DEPTH_COUNT=$(ls -1 "$RGB_DEPTH_ROOT" | wc -l)
    echo "  → Found $RGB_DEPTH_COUNT scene directories"
else
    check_fail "RGB/Depth root NOT FOUND: $RGB_DEPTH_ROOT"
    RGB_DEPTH_COUNT=0
fi

if [ -d "$PLANE_LABEL_ROOT" ]; then
    check_pass "Plane label root exists: $PLANE_LABEL_ROOT"
    PLANE_COUNT=$(ls -1 "$PLANE_LABEL_ROOT" | wc -l)
    echo "  → Found $PLANE_COUNT scene directories"
else
    check_fail "Plane label root NOT FOUND: $PLANE_LABEL_ROOT"
    PLANE_COUNT=0
fi

if [ -d "$INTRINSICS_ROOT" ]; then
    check_pass "Intrinsics root exists: $INTRINSICS_ROOT"
    INTRINSICS_COUNT=$(ls -1 "$INTRINSICS_ROOT" | wc -l)
    echo "  → Found $INTRINSICS_COUNT scene directories"
else
    check_fail "Intrinsics root NOT FOUND: $INTRINSICS_ROOT"
    INTRINSICS_COUNT=0
fi

if [ -d "$SPLIT_TXT_DIR" ]; then
    check_pass "Split txt dir exists: $SPLIT_TXT_DIR"
else
    check_fail "Split txt dir NOT FOUND: $SPLIT_TXT_DIR"
fi

################################################################################
# CHECK 2: Split Files
################################################################################

print_header "CHECK 2: Split Files Validation"

for SPLIT in train val test; do
    SPLIT_FILE="$SPLIT_TXT_DIR/${SPLIT}.txt"
    if [ -f "$SPLIT_FILE" ]; then
        SCENE_COUNT=$(wc -l < "$SPLIT_FILE")
        check_pass "$SPLIT.txt exists with $SCENE_COUNT scenes"
        echo "  → First 3 scenes: $(head -3 "$SPLIT_FILE" | tr '\n' ' ')"
    else
        check_fail "$SPLIT.txt NOT FOUND"
    fi
done

################################################################################
# CHECK 3: Sample Scene Structure
################################################################################

print_header "CHECK 3: Sample Scene Structure"

# Get first scene from val split
if [ -f "$SPLIT_TXT_DIR/val.txt" ]; then
    SAMPLE_SCENE=$(head -1 "$SPLIT_TXT_DIR/val.txt")
    echo "Testing with sample scene: $SAMPLE_SCENE"

    print_subheader "RGB/Depth files:"
    if [ -d "$RGB_DEPTH_ROOT/$SAMPLE_SCENE" ]; then
        MERGED_FILES=$(ls "$RGB_DEPTH_ROOT/$SAMPLE_SCENE"/*_merged.h5 2>/dev/null | wc -l || echo 0)
        if [ "$MERGED_FILES" -gt 0 ]; then
            check_pass "Found $MERGED_FILES merged HDF5 files"
            echo "  Files:"
            ls -lh "$RGB_DEPTH_ROOT/$SAMPLE_SCENE"/*_merged.h5 | awk '{print "    " $9 " (" $5 ")"}'
        else
            check_fail "No *_merged.h5 files found in $RGB_DEPTH_ROOT/$SAMPLE_SCENE"
        fi
    else
        check_fail "Scene directory not found: $RGB_DEPTH_ROOT/$SAMPLE_SCENE"
    fi

    print_subheader "Plane label files:"
    if [ -d "$PLANE_LABEL_ROOT/$SAMPLE_SCENE" ]; then
        PLANE_FILES=$(ls "$PLANE_LABEL_ROOT/$SAMPLE_SCENE"/rendered_planes_*.h5 2>/dev/null | wc -l || echo 0)
        if [ "$PLANE_FILES" -gt 0 ]; then
            check_pass "Found $PLANE_FILES plane HDF5 files"
            echo "  Files:"
            ls -lh "$PLANE_LABEL_ROOT/$SAMPLE_SCENE"/rendered_planes_*.h5 | awk '{print "    " $9 " (" $5 ")"}'
        else
            check_fail "No rendered_planes_*.h5 files found in $PLANE_LABEL_ROOT/$SAMPLE_SCENE"
        fi
    else
        check_fail "Scene directory not found: $PLANE_LABEL_ROOT/$SAMPLE_SCENE"
    fi

    print_subheader "Intrinsics files:"
    if [ -d "$INTRINSICS_ROOT/$SAMPLE_SCENE" ]; then
        INTRINSIC_FILES=$(ls "$INTRINSICS_ROOT/$SAMPLE_SCENE"/*_intrinsics.json 2>/dev/null | wc -l || echo 0)
        if [ "$INTRINSIC_FILES" -gt 0 ]; then
            check_pass "Found $INTRINSIC_FILES intrinsics JSON files"
            echo "  Files:"
            ls -lh "$INTRINSICS_ROOT/$SAMPLE_SCENE"/*_intrinsics.json | awk '{print "    " $9 " (" $5 ")"}'
        else
            check_fail "No *_intrinsics.json files found in $INTRINSICS_ROOT/$SAMPLE_SCENE"
        fi
    else
        check_fail "Scene directory not found: $INTRINSICS_ROOT/$SAMPLE_SCENE"
    fi
else
    check_fail "Cannot test sample scene - val.txt not found"
    SAMPLE_SCENE=""
fi

################################################################################
# CHECK 4: HDF5 File Format Verification
################################################################################

print_header "CHECK 4: HDF5 File Format Verification"

if [ -n "$SAMPLE_SCENE" ]; then
    # Check merged file
    MERGED_FILE=$(ls "$RGB_DEPTH_ROOT/$SAMPLE_SCENE"/*_merged.h5 2>/dev/null | head -1)
    if [ -f "$MERGED_FILE" ]; then
        echo "Checking: $MERGED_FILE"
        python3 << EOF
import h5py
import sys

try:
    with h5py.File("$MERGED_FILE", "r") as f:
        keys = list(f.keys())
        print(f"  Keys found: {keys}")

        has_rgb = "rgb" in keys
        has_depth = "depth" in keys

        if has_rgb:
            print(f"  ✓ 'rgb' dataset found, shape: {f['rgb'].shape}, dtype: {f['rgb'].dtype}")
        else:
            print(f"  ✗ 'rgb' dataset NOT FOUND")
            sys.exit(1)

        if has_depth:
            print(f"  ✓ 'depth' dataset found, shape: {f['depth'].shape}, dtype: {f['depth'].dtype}")
        else:
            print(f"  ✗ 'depth' dataset NOT FOUND")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Error reading HDF5: {e}")
    sys.exit(1)
EOF
        if [ $? -eq 0 ]; then
            check_pass "Merged HDF5 file has correct structure"
        else
            check_fail "Merged HDF5 file has incorrect structure"
        fi
    else
        check_warn "No merged file found to check"
    fi

    # Check plane file
    PLANE_FILE=$(ls "$PLANE_LABEL_ROOT/$SAMPLE_SCENE"/rendered_planes_*.h5 2>/dev/null | head -1)
    if [ -f "$PLANE_FILE" ]; then
        echo "Checking: $PLANE_FILE"
        python3 << EOF
import h5py
import sys

try:
    with h5py.File("$PLANE_FILE", "r") as f:
        keys = list(f.keys())
        print(f"  Keys found: {keys}")

        has_planes = "planes" in keys
        has_frame_ids = "frame_ids" in keys

        if has_planes:
            print(f"  ✓ 'planes' dataset found, shape: {f['planes'].shape}, dtype: {f['planes'].dtype}")
        else:
            print(f"  ✗ 'planes' dataset NOT FOUND")
            sys.exit(1)

        if has_frame_ids:
            print(f"  ✓ 'frame_ids' dataset found, shape: {f['frame_ids'].shape}, dtype: {f['frame_ids'].dtype}")
            print(f"    Sample frame IDs: {[fid.decode('utf-8') if isinstance(fid, bytes) else str(fid) for fid in f['frame_ids'][:3]]}")
        else:
            print(f"  ✗ 'frame_ids' dataset NOT FOUND")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Error reading HDF5: {e}")
    sys.exit(1)
EOF
        if [ $? -eq 0 ]; then
            check_pass "Plane HDF5 file has correct structure"
        else
            check_fail "Plane HDF5 file has incorrect structure"
        fi
    else
        check_warn "No plane file found to check"
    fi

    # Check intrinsics file
    INTRINSICS_FILE=$(ls "$INTRINSICS_ROOT/$SAMPLE_SCENE"/*_intrinsics.json 2>/dev/null | head -1)
    if [ -f "$INTRINSICS_FILE" ]; then
        echo "Checking: $INTRINSICS_FILE"
        python3 << EOF
import json
import sys

try:
    with open("$INTRINSICS_FILE", "r") as f:
        data = json.load(f)

    if "K" in data:
        K = data["K"]
        print(f"  ✓ 'K' matrix found: {K}")
        if len(K) == 3 and all(len(row) == 3 for row in K):
            print(f"  ✓ K is 3x3 matrix")
        else:
            print(f"  ✗ K is not 3x3 matrix")
            sys.exit(1)
    else:
        print(f"  ✗ 'K' matrix NOT FOUND in JSON")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Error reading JSON: {e}")
    sys.exit(1)
EOF
        if [ $? -eq 0 ]; then
            check_pass "Intrinsics JSON file has correct structure"
        else
            check_fail "Intrinsics JSON file has incorrect structure"
        fi
    else
        check_warn "No intrinsics file found to check"
    fi
else
    check_warn "Skipping HDF5 format check - no sample scene available"
fi

################################################################################
# CHECK 5: Dataset Loading Test
################################################################################

print_header "CHECK 5: Python Dataset Loading Test"

echo "Running Python verification script..."
python3 planamono/shared/datasets/verify_hypersim_splits.py

PYTHON_EXIT=$?
if [ $PYTHON_EXIT -eq 0 ]; then
    check_pass "Python dataset verification completed"
else
    check_fail "Python dataset verification failed with exit code $PYTHON_EXIT"
fi

################################################################################
# SUMMARY
################################################################################

print_header "VERIFICATION SUMMARY"

echo ""
echo "Total checks performed: $TOTAL_CHECKS"
echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
echo -e "${YELLOW}Warnings: $WARNING_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Your Hypersim dataset is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Run evaluation: python planamono/evaluation/quantitative/evaluate_hypersim_fast.py"
    echo "  2. Check results in the output directory"
    exit 0
else
    echo -e "${RED}${BOLD}✗ SOME CHECKS FAILED${NC}"
    echo ""
    echo "Please review the errors above and:"
    echo "  1. Verify your data paths are correct"
    echo "  2. Check that files follow the expected naming convention"
    echo "  3. See docs/hypersim_dataset_setup.md for troubleshooting"
    exit 1
fi
