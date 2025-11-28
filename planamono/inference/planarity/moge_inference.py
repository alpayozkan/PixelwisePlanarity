#!/usr/bin/env python3
"""
MoGe Planarity Inference Module

Provides the MoGePlanarityInference class for running planarity prediction
using trained MoGe 4-head models.
"""
import os
import sys
from pathlib import Path

# Add project root to path
# script_dir = Path(__file__).resolve().parent
# project_root = script_dir.parent.parent
# sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from planamono.moge.moge.model.v2 import MoGeModel


class MoGePlanarityInference:
    """Class for performing inference with trained MoGe 4-head planarity model."""

    def __init__(self, model_path, model_size='large', device='cuda', cache_dir=None):
        """
        Args:
            model_path: Path to trained model checkpoint (.pt file)
            model_size: Model size ('small', 'middle', 'large')
            device: Device for inference ('cuda' or 'cpu')
            cache_dir: Directory for caching pretrained models (or set MOGE_CACHE_DIR env var)
        """
        # Get cache directory from arg or env var
        if cache_dir is None:
            cache_dir = os.environ.get("MOGE_CACHE_DIR")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load the trained model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Model name mapping
        model_names = {
            'large': "Ruicheng/moge-2-vitl-normal",
            'middle': "Ruicheng/moge-2-vitb-normal",
            'small': "Ruicheng/moge-2-vits-normal"
        }

        if model_size not in model_names:
            raise ValueError(f"model_size must be one of {list(model_names.keys())}, got '{model_size}'")

        # Load pretrained model
        model_name = model_names[model_size]
        print(f"Loading pretrained model: {model_name}")
        if cache_dir:
            self.model = MoGeModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)
        else:
            self.model = MoGeModel.from_pretrained(model_name).to(device)

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
        from moge.moge.model.v2 import normalized_view_plane_uv
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