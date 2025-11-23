import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob
import re

from PIL import Image
import cv2

from shared.segmentation import compute_vectorized_planar_segments_v1, filter_small_segments, remove_small_components
from shared.utils import depth_to_normal_remi, extract_zdepth, visualize_top_components_v1

import argparse
import torch
import torch.nn.functional as F
from natsort import natsorted

# External dependency - MoGe model
try:
    from moge.model.v2 import MoGeModel
except ImportError:
    print("[WARN] MoGe not found. Install from: https://github.com/microsoft/MoGe")
    MoGeModel = None


class MoGePlanarityInference:
    """Class for performing inference with trained MoGe 4-head planarity model."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: Device for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model - need to use the same base model that was used for training
        # self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)

        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal", cache_dir="/cluster/scratch/aoezkan/MoGe/checkpoints/").to(device)      

        # self.model = MoGeModelPlanarity.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        
        # Load the state dict (this should include the planarity head)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Verify planarity head exists
        if not hasattr(self.model, 'planarity_head'):
            raise ValueError("Loaded model does not have planarity_head! Make sure you're loading a 4-head model.")
        
        print("✓ Planarity head found in model")
        
        # Print model info if available
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint:
            print(f"Final validation loss: {checkpoint['val_loss']:.4f}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def preprocess_image(self, image_path, target_height=512, target_width=768):
        """
        Preprocess input image for inference.
        
        Args:
            image_path: Path to input image
            target_height: Target height for resizing
            target_width: Target width for resizing
        
        Returns:
            Preprocessed image tensor
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
        
        # Store original dimensions
        original_h, original_w = image.shape[:2]
        
        # Resize to target dimensions
        image_resized = cv2.resize(image, (target_width, target_height))
        
        # Convert to tensor and normalize to [0, 1]
        image_tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor, (original_h, original_w)
    
    def _manual_forward(self, images, num_tokens=1024):
        """Manual forward pass to get planarity predictions."""
        batch_size, _, img_h, img_w = images.shape
        device, dtype = images.device, images.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = int((num_tokens / aspect_ratio) ** 0.5), int((num_tokens * aspect_ratio) ** 0.5)
        num_tokens = base_h * base_w

        # Backbone encoding
        features, cls_token = self.model.encoder(images, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        from moge.model.v2 import normalized_view_plane_uv
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        features = self.model.neck(features)

        # Get all head outputs
        outputs = {}
        
        # Planarity head (our main target)
        if hasattr(self.model, 'planarity_head'):
            planarity_raw_logits = self.model.planarity_head(features)[-1]
            planarity_raw_logits = F.interpolate(planarity_raw_logits, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False)
            outputs['planarity_logits'] = planarity_raw_logits
            outputs['planarity'] = torch.sigmoid(planarity_raw_logits)
        
        # Mask head
        if hasattr(self.model, 'mask_head'):
            mask_raw = self.model.mask_head(features)[-1]
            mask_raw = F.interpolate(mask_raw, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False)
            outputs['mask_logits'] = mask_raw
            outputs['mask'] = torch.sigmoid(mask_raw)
        
        # Normal head
        if hasattr(self.model, 'normal_head'):
            normal_raw = self.model.normal_head(features)[-1]
            normal_raw = F.interpolate(normal_raw, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False)
            outputs['normal'] = normal_raw
        
        # Points head
        if hasattr(self.model, 'points_head'):
            points_raw = self.model.points_head(features)[-1]
            points_raw = F.interpolate(points_raw, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False)
            outputs['points'] = points_raw
        
        return outputs
    
    def predict(self, image_path, num_tokens=1024, return_all_heads=False):
        """
        Perform planarity prediction on input image.
        
        Args:
            image_path: Path to input image or numpy array
            num_tokens: Number of tokens for the model
            return_all_heads: If True, return outputs from all heads
        
        Returns:
            Dictionary containing predictions
        """
        with torch.no_grad():
            # Preprocess image
            image_tensor, original_size = self.preprocess_image(image_path)
            
            # Forward pass with autocast for efficiency
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                try:
                    # Try built-in forward first
                    outputs = self.model(image_tensor, num_tokens=num_tokens)
                    if 'planarity' not in outputs:
                        # Fall back to manual forward
                        outputs = self._manual_forward(image_tensor, num_tokens)
                except:
                    # Fall back to manual forward
                    outputs = self._manual_forward(image_tensor, num_tokens)
            
            # Extract results
            results = {}
            
            # Main planarity prediction
            if 'planarity' in outputs:
                planarity_prob = outputs['planarity'].squeeze().cpu().numpy()
                planarity_binary = (planarity_prob > 0.5).astype(np.uint8)
                
                results['planarity_probability'] = planarity_prob
                results['planarity_binary'] = planarity_binary
                
                # Resize back to original image size
                if original_size != planarity_prob.shape:
                    planarity_prob_resized = cv2.resize(planarity_prob, (original_size[1], original_size[0]))
                    planarity_binary_resized = cv2.resize(planarity_binary, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
                    
                    results['planarity_probability_full'] = planarity_prob_resized
                    results['planarity_binary_full'] = planarity_binary_resized
            
            # Additional heads if requested
            if return_all_heads:
                if 'mask' in outputs:
                    mask_prob = outputs['mask'].squeeze().cpu().numpy()
                    results['mask'] = mask_prob
                
                if 'normal' in outputs:
                    normal = outputs['normal'].squeeze().cpu().numpy()
                    if normal.ndim == 3:  # CHW format
                        normal = normal.transpose(1, 2, 0)  # Convert to HWC
                    results['normal'] = normal
                
                if 'points' in outputs:
                    points = outputs['points'].squeeze().cpu().numpy()
                    results['points'] = points
            
            return results

    
    def predict_batch(self, image_paths, num_tokens=1024, batch_size=4):
        """
        Perform batch prediction on multiple images.
        
        Args:
            image_paths: List of image paths
            num_tokens: Number of tokens for the model
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for path in batch_paths:
                result = self.predict(path, num_tokens)
                result['image_path'] = path
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None, show_overlay=True, return_all_heads=False):
        """
        Visualize planarity prediction results.
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization (if None, display only)
            show_overlay: Whether to show overlay visualization
            return_all_heads: Whether to include other head outputs
        
        Returns:
            Matplotlib figure
        """
        # Get prediction
        results = self.predict(image_path, return_all_heads=return_all_heads)
        
        # Load original image
        original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # Determine number of subplots
        num_cols = 4 if return_all_heads else 3
        if show_overlay:
            num_cols += 1
        
        fig, axes = plt.subplots(1, num_cols, figsize=(5*num_cols, 5))
        if num_cols == 1:
            axes = [axes]
        
        col_idx = 0
        
        # Original image
        axes[col_idx].imshow(original_image)
        axes[col_idx].set_title('Original Image')
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Planarity probability
        if 'planarity_probability_full' in results:
            planarity_vis = results['planarity_probability_full']
        else:
            planarity_vis = results['planarity_probability']
        
        axes[col_idx].imshow(planarity_vis, cmap='gray', vmin=0, vmax=1)
        axes[col_idx].set_title('Planarity Probability')
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Binary planarity
        if 'planarity_binary_full' in results:
            binary_vis = results['planarity_binary_full']
        else:
            binary_vis = results['planarity_binary']
        
        axes[col_idx].imshow(binary_vis, cmap='gray', vmin=0, vmax=1)
        axes[col_idx].set_title('Binary Planarity')
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Overlay visualization
        if show_overlay:
            overlay = original_image.copy()
            if 'planarity_binary_full' in results:
                binary_mask = results['planarity_binary_full']
            else:
                binary_mask = cv2.resize(results['planarity_binary'].astype(np.uint8), 
                                       (original_image.shape[1], original_image.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST)
            
            # Add blue overlay for planar regions
            overlay[:, :, 2] = np.maximum(overlay[:, :, 2], binary_mask * 200)
            
            axes[col_idx].imshow(overlay)
            axes[col_idx].set_title('Planarity Overlay (Blue)')
            axes[col_idx].axis('off')
            col_idx += 1
        
        # Additional heads if requested
        if return_all_heads:
            if 'mask' in results:
                axes[col_idx].imshow(results['mask'], cmap='gray', vmin=0, vmax=1)
                axes[col_idx].set_title('Mask')
                axes[col_idx].axis('off')
                col_idx += 1
            
            if 'normal' in results:
                normal_vis = results['normal']
                if normal_vis.ndim == 3:
                    # Normalize normal map for visualization
                    normal_vis = (normal_vis + 1) / 2  # Convert from [-1,1] to [0,1]
                axes[col_idx].imshow(normal_vis)
                axes[col_idx].set_title('Normal Map')
                axes[col_idx].axis('off')
                col_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig

parser = argparse.ArgumentParser(description="MoGe 4-head planarity inference")

# Model arguments
parser.add_argument("--model_path", type=str, required=True,
                   help="Path to trained model checkpoint (.pt file)")
parser.add_argument("--device", type=str, default="cuda",
                   help="Device for inference (cuda/cpu)")

# Input arguments
# parser.add_argument("--input", type=str, required=True,
#                    help="Input image path or directory containing images")
# parser.add_argument("--output_dir", type=str, default="./inference_results",
#                    help="Output directory for results")

# Inference arguments
parser.add_argument("--num_tokens", type=int, default=1024,
                   help="Number of tokens for the model")
parser.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for inference")
parser.add_argument("--save_raw", action="store_true",
                   help="Save raw probability maps as .npy files")
parser.add_argument("--save_binary", action="store_true",
                   help="Save binary masks as .png files")
parser.add_argument("--save_visualization", action="store_true",
                   help="Save visualization images")
parser.add_argument("--return_all_heads", action="store_true",
                   help="Include outputs from all heads (mask, normal, points)")

args = parser.parse_args(args=[
    "--model_path", "/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt",
    "--device", "cuda",
])
# args = parser.parse_args(args=[
#     "--model_path", "/cluster/scratch/aoezkan/MoGe/checkpoints/final_planarity_4heads_model.pt",
#     "--device", "cuda",
# ])

print("MoGe 4-Head Planarity Inference")
print("=" * 40)
print(f"Model: {args.model_path}")
# print(f"Input: {args.input}")
# print(f"Output: {args.output_dir}")
print(f"Device: {args.device}")
print("=" * 40)

inference_model = MoGePlanarityInference(args.model_path, device=args.device)


inference_model.model.encoder.use_memory_efficient_attention = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
inference_model.model = inference_model.model.half()  # or .to(dtype=torch.float16)
inference_model.model.encoder.enable_pytorch_native_sdpa()


# Update for scannet: camera + depth separate, dont violate the resolution etc
#########################################
# fx = 886.810
# fy = 886.810
# cx = 512.0
# cy = 384.0

apply_billateral=True
threshold_planarity = 0.6
neighbor_match_count_thresh = 24
normal_threshold_deg = 10.0
depth_threshold = 0.05



# Path to the normal map image
# normal_map_path = '/work/courses/3dv/32/hypersim_dataset/dataset/ai_001_001/images/scene_cam_00_final_preview/frame.0000.color.jpg'
# labels_base_dir = '/work/courses/3dv/32/data/dbscan_out_v1/planarity_new_method/'

root_dir = '/cluster/scratch/aoezkan/dataset/scannet_new/scans'
result_dir = '/cluster/scratch/aoezkan/results/scannet/'


scene_id_list = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
scene_id_list = natsorted(scene_id_list)
assert len(scene_id_list) > 0, "No scene IDs found in the root directory."

# print("Scene IDs:", scene_id_list)
# print(len(scene_id_list))

# scene_id = 'scene0000_00'
# scene_id = scene_id_list[0]
for scene_id in tqdm(scene_id_list):
    # depthany_dir = os.path.join(root_dir, scene_id, 'depthany')
    # planarity_dir = os.path.join(root_dir, scene_id, 'planarity_depthany')
    # normal_out_dir = os.path.join(root_dir, scene_id, 'stable_normal')
    
    color_dir = os.path.join(root_dir, scene_id, 'color')
    image_list = natsorted(glob.glob(os.path.join(color_dir, '*.jpg')))
    # print(f"Found {len(image_list)} images in {color_dir}")
    # print(image_list[:1])  # Show first 5 images

    seg_dir = os.path.join(result_dir, 'moge', 'seg_pred', scene_id)
    os.makedirs(seg_dir, exist_ok=True)
    segvis_dir = os.path.join(result_dir, 'moge', 'seg_vis', scene_id)
    os.makedirs(segvis_dir, exist_ok=True)
    
    # for image_path in image_list:
    for idx in range(0, len(image_list), 50):
        image_path = image_list[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # depthany_save_path = os.path.join(depthany_dir, f"{base_name}_depthany.npy")
        # normal_save_path = os.path.join(normal_out_dir, f"{base_name}_normal.npy")
        # planarity_save_path = os.path.join(planarity_dir, f"{base_name}_planarity_depthany.npy")
        
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        res = inference_model.predict(image_path, num_tokens=1024, return_all_heads=True)
        
        depth = res['points'][:,:,2]
        normal = np.transpose(res['normal'], (2,0,1))
        # planarity = res['planarity_probability']
        # planarity = res['planarity_binary']
        planarity = res['planarity_probability']
        # planarity_prob = res['planarity_probability']

        # depth = np.load(depthany_save_path)
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
        # planarity = np.load(planarity_save_path)
        # normal = np.load(normal_save_path)

        depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        normal = cv2.resize(normal.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        planarity = cv2.resize(planarity.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        planarity_mask = planarity>threshold_planarity
        planarity_mask = planarity_mask.astype(np.int16)

        assert img_np.shape[:2] == depth.shape[:2] == normal.shape[:2] == planarity.shape[:2], "All inputs must have the same resolution"

        # calculate segmentation
        normal_threshold_rad = np.deg2rad(normal_threshold_deg)
        labels, n_components = compute_vectorized_planar_segments_v1(
            planarity_mask, normal, depth,
            normal_threshold_rad, depth_threshold,
            neighbor_match_count_thresh=neighbor_match_count_thresh
        )
        segmentation = labels.copy()
        filtered_segmentation = remove_small_components(segmentation, min_size=500)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        seg_save_path = os.path.join(seg_dir, f"{base_name}_seg_pred.npy")
        np.save(seg_save_path, filtered_segmentation)

        segvis_save_path = os.path.join(segvis_dir, f"{base_name}_seg_vis.png")
        

        fig, axs = plt.subplots(1, 7, figsize=(16, 4))
        fig.suptitle(f"Scannet Image: {scene_id}_{base_name}", fontsize=16)
        
        im0 = axs[0].imshow(img)
        axs[0].set_title("Image")
        axs[0].axis('off')
        fig.colorbar(im0, ax=axs[0], orientation='horizontal')
        
        im1 = axs[1].imshow(depth)
        axs[1].set_title("Depth Moge")
        axs[1].axis('off')
        fig.colorbar(im1, ax=axs[1], orientation='horizontal')
        
        im2 = axs[2].imshow(normal)
        axs[2].set_title("Normal Moge")
        axs[2].axis('off')
        fig.colorbar(im2, ax=axs[2], orientation='horizontal')
        
        im3 = axs[3].imshow(planarity)
        axs[3].set_title("Moge Planarity")
        axs[3].axis('off')
        fig.colorbar(im3, ax=axs[3], orientation='horizontal')
        
        im3 = axs[4].imshow(planarity_mask)
        axs[4].set_title("Moge Planarity Thresh")
        axs[4].axis('off')
        fig.colorbar(im3, ax=axs[4], orientation='horizontal')
        
        
        res_seg = filtered_segmentation.copy()
        n = len(np.unique(res_seg))
        n1 = min(10, n)
        n2 = max(n//2, 1)
        seg_top1 = visualize_top_components_v1(
            res_seg, k=n1, return_colors=True
        )
        seg_top2 = visualize_top_components_v1(
            res_seg, k=n2, return_colors=True
        )
        
        axs[5].imshow(seg_top1)
        axs[5].set_title("Seg: Top-{}".format(n1))
        axs[5].axis('off')
        
        axs[6].imshow(seg_top2)
        axs[6].set_title("Seg: Top-{}".format(n2))
        axs[6].axis('off')
        
        plt.tight_layout()
        # plt.show()
        fig.savefig(segvis_save_path)
        # fig.savefig(f"{base_name}_visualization.png")
        plt.close(fig)
        # break
    # break