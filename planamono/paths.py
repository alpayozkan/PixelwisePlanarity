# here we list paths for dataset and repo

# repo_path = '/cluster/home/aoezkan/planeseg/planamono/planamono'
repo_path = '/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/'

# dataset dirs (original)
hypersim_path = '/cluster/project/cvg/Shared_datasets/Hypersim'
scannetpp_path = '/cluster/project/cvg/Shared_datasets/scannet++'
scannetppv2_path = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'

# our plane gt dataset dirs (3D mesh plane extraction)
hypersim_plane_path = '/cluster/scratch/ayavuz/dataset/Hypersim_ours'
scannetpp_plane_path = '/cluster/scratch/ayavuz/dataset/plane_ours_gt'

# ============ Hypersim processed data ============
# depth + semantics + RGB (merged)
hypersim_merged_path = '/cluster/scratch/ayavuz/dataset/Hypersim_merged'
# camera parameters
# hypersim_params_path = '/cluster/scratch/ayavuz/dataset/Hypersim_params'
hypersim_params_path = '/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params'
# plane segmentation labels (rendered)
hypersim_rendered_path = '/cluster/scratch/ayavuz/dataset/Hypersim_rendered'

# ============ ScanNet++ rendered data ============
scannetpp_rend_plane_path = '/cluster/scratch/aoezkan/planeseg/dataset/scannetpp'

# Legacy paths (for backward compatibility)
hypersim_rend_plane_path = '/cluster/scratch/aoezkan/planeseg/dataset/Hypersim'