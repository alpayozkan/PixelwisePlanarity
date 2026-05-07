# here we list paths for dataset and repo

# repo_path = '/cluster/home/aoezkan/planeseg/planamono/planamono'
repo_path = '/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/'

# dataset dirs (original)
hypersim_path = '/cluster/project/cvg/Shared_datasets/Hypersim'
scannetpp_path = '/cluster/project/cvg/Shared_datasets/scannet++'
scannetppv2_path = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'

# our plane gt dataset dirs (3D mesh plane extraction)
hypersim_plane_path = '/cluster/scratch/ayavuz/dataset/Hypersim_ours'
# scannetpp_plane_path = '/cluster/scratch/ayavuz/dataset/plane_ours_gt'
scannetpp_plane_path = '/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp/'

# ============ Hypersim processed data ============
# New unified dataset (RGB + depth + rendered plane labels under one root)
hypersim_merged_path = '/cluster/scratch/aoezkan/planeseg/dataset/hypersim'
hypersim_rendered_path = '/cluster/scratch/aoezkan/planeseg/dataset/hypersim'
# camera parameters (not in the unified dataset, kept separately)
# hypersim_params_path = '/cluster/scratch/ayavuz/dataset/Hypersim_params'
hypersim_params_path = '/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params'

# Old Hypersim paths (buggy plane_id=0 collision in rendered labels)
# hypersim_merged_path = '/cluster/scratch/ayavuz/dataset/Hypersim_merged'
# hypersim_rendered_path = '/cluster/scratch/ayavuz/dataset/Hypersim_rendered'

# ============ ScanNet++ rendered data ============
scannetpp_rend_plane_path = '/cluster/scratch/aoezkan/planeseg/dataset/scannetpp'

# Legacy paths (for backward compatibility)
hypersim_rend_plane_path = '/cluster/scratch/aoezkan/planeseg/dataset/Hypersim'

# ============ VKITTI2 and Synthia ============
# vkitti2_path = '/cluster/scratch/aoezkan/planeseg/dataset/vkitti2'
# synthia_path = '/cluster/scratch/aoezkan/planeseg/dataset/synthia'
vkitti2_path = '/cluster/scratch/ayavuz/dataset/vkitti2_planes'
synthia_path = '/cluster/scratch/ayavuz/dataset/synthia_planes'

# ============ NYU-v2 and 7-Scenes (ZeroPlane "_d2" NPZ format) ============
nyuv2_path = '/cluster/scratch/aoezkan/planeseg/dataset/nyuv2_plane'
sevenscenes_path = '/cluster/scratch/aoezkan/planeseg/dataset/sevenscenes_plane'