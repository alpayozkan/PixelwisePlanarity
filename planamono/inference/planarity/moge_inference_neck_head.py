"""
Inference classes for neck_head and proj_neck_head planarity architectures.

These architectures use separate (copied) neck/head modules for planarity,
while the frozen MoGe model handles depth, normals, and mask.

- MoGePlanarityNeckHeadInference: frozen encoder -> trainable neck + head
- MoGePlanarityProjNeckHeadInference: frozen backbone -> trainable projections + neck + head
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from planamono.moge.moge.model.v2 import MoGeModel
from planamono.moge.moge.utils.geometry_torch import normalized_view_plane_uv


def _modify_head_last_layer(head, device):
    """Change last Conv2d in head from 3 output channels to 1."""
    last_conv = None
    for name, module in head.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)

    if last_conv is None:
        raise ValueError("Could not find last Conv2d in head!")

    name, old_conv = last_conv
    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=1,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    ).to(device)

    parent = head
    parts = name.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_conv)


def _build_features_list(encoder_features, images, base_h, base_w):
    """Build 5-level feature list with UV coords (same as MoGeModel.forward)."""
    B, _, img_h, img_w = images.shape
    aspect_ratio = img_w / img_h
    dtype, device = images.dtype, images.device

    features_list = [encoder_features, None, None, None, None]
    for level in range(5):
        uv = normalized_view_plane_uv(
            width=base_w * 2 ** level,
            height=base_h * 2 ** level,
            aspect_ratio=aspect_ratio,
            dtype=dtype,
            device=device,
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        if features_list[level] is None:
            features_list[level] = uv
        else:
            features_list[level] = torch.cat([features_list[level], uv], dim=1)
    return features_list


def _compute_base_hw(images, num_tokens):
    _, _, img_h, img_w = images.shape
    aspect_ratio = img_w / img_h
    base_h = int((num_tokens / aspect_ratio) ** 0.5)
    base_w = int((num_tokens * aspect_ratio) ** 0.5)
    return base_h, base_w


class MoGePlanarityNeckHeadInference:
    """Inference for neck_head architecture: frozen encoder -> separate neck + head for planarity."""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load frozen base MoGe model
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Deep-copy neck and normal_head for planarity
        self.planarity_neck = copy.deepcopy(self.model.neck).to(self.device)
        self.planarity_head = copy.deepcopy(self.model.normal_head).to(self.device)
        _modify_head_last_layer(self.planarity_head, self.device)

        # Load trained weights
        self.planarity_neck.load_state_dict(checkpoint['planarity_neck_state_dict'])
        self.planarity_head.load_state_dict(checkpoint['planarity_head_state_dict'])
        self.planarity_neck.eval()
        self.planarity_head.eval()

        # Cast to float32
        self.model = self.model.float().to(self.device)
        self.planarity_neck = self.planarity_neck.float().to(self.device)
        self.planarity_head = self.planarity_head.float().to(self.device)

        print("Model loaded successfully!")
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    def _forward(self, images, num_tokens=1024):
        B, _, img_h, img_w = images.shape
        base_h, base_w = _compute_base_hw(images, num_tokens)

        # Frozen encoder
        with torch.no_grad():
            encoder_features, _ = self.model.encoder(images, base_h, base_w, return_class_token=True)

        features_list = _build_features_list(encoder_features, images, base_h, base_w)

        # Planarity: separate neck + head
        planarity_neck_out = self.planarity_neck(features_list)
        planarity_logits = self.planarity_head(planarity_neck_out)[-1]
        planarity_logits = F.interpolate(planarity_logits, (img_h, img_w),
                                         mode='bilinear', align_corners=False)

        outputs = {
            'planarity_logits': planarity_logits,
            'planarity': torch.sigmoid(planarity_logits),
        }

        # Other heads: frozen model's original neck + heads
        # Apply same post-processing as MoGeModel.forward():
        #   points/normal: permute (B,C,H,W) -> (B,H,W,C), normalize normals, remap points
        #   mask: squeeze + sigmoid
        with torch.no_grad():
            frozen_neck_out = self.model.neck(features_list)

            if hasattr(self.model, 'points_head'):
                points_raw = self.model.points_head(frozen_neck_out)[-1]
                points_raw = F.interpolate(points_raw, (img_h, img_w),
                                           mode='bilinear', align_corners=False)
                points_raw = points_raw.permute(0, 2, 3, 1)
                outputs['points'] = self.model._remap_points(points_raw)

            if hasattr(self.model, 'normal_head'):
                normal_raw = self.model.normal_head(frozen_neck_out)[-1]
                normal_raw = F.interpolate(normal_raw, (img_h, img_w),
                                           mode='bilinear', align_corners=False)
                normal_raw = normal_raw.permute(0, 2, 3, 1)
                outputs['normal'] = F.normalize(normal_raw, dim=-1)

            if hasattr(self.model, 'mask_head'):
                mask_raw = self.model.mask_head(frozen_neck_out)[-1]
                mask_raw = F.interpolate(mask_raw, (img_h, img_w),
                                         mode='bilinear', align_corners=False)
                outputs['mask'] = mask_raw.squeeze(1).sigmoid()

        return outputs

    def preprocess_image(self, image_path, target_height=512, target_width=768):
        if isinstance(image_path, str):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        original_h, original_w = image.shape[:2]
        image_resized = cv2.resize(image, (target_width, target_height))
        image_tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor, (original_h, original_w)

    def preprocess_images(self, image_paths, target_height=512, target_width=768):
        images = []
        original_sizes = []
        for p in image_paths:
            if isinstance(p, str):
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            else:
                img = p
            h, w = img.shape[:2]
            original_sizes.append((h, w))
            img_resized = cv2.resize(img, (target_width, target_height))
            img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
            images.append(img_tensor)
        images = torch.stack(images, dim=0).to(self.device)
        return images, original_sizes

    def predict(self, image_path, num_tokens=1024, return_all_heads=False):
        with torch.no_grad():
            image_tensor, original_size = self.preprocess_image(image_path)
            outputs = self._forward(image_tensor, num_tokens)

            results = {}
            if 'planarity' in outputs:
                planarity_prob = outputs['planarity'].squeeze().cpu().numpy()
                planarity_binary = (planarity_prob > 0.5).astype(np.uint8)
                results['planarity_probability'] = planarity_prob
                results['planarity_binary'] = planarity_binary

                if original_size != planarity_prob.shape:
                    results['planarity_probability_full'] = cv2.resize(
                        planarity_prob, (original_size[1], original_size[0]))
                    results['planarity_binary_full'] = cv2.resize(
                        planarity_binary, (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_NEAREST)

            if return_all_heads:
                if 'mask' in outputs:
                    results['mask'] = outputs['mask'].squeeze().cpu().numpy()
                if 'normal' in outputs:
                    # MoGe v2 format: (B, H, W, 3), squeeze gives (H, W, 3)
                    results['normal'] = outputs['normal'].squeeze(0).cpu().numpy()
                if 'points' in outputs:
                    # MoGe v2 format: (B, H, W, 3), squeeze gives (H, W, 3)
                    results['points'] = outputs['points'].squeeze(0).cpu().numpy()

            return results

    def predict_batch_fast(self, image_paths, num_tokens=1024, return_all_heads=False):
        with torch.no_grad():
            images, original_sizes = self.preprocess_images(image_paths)
            outputs = self._forward(images, num_tokens)

            B = images.shape[0]
            results = []
            for i in range(B):
                res = {}
                planarity = outputs['planarity'][i].squeeze().cpu().numpy()
                planarity_bin = (planarity > 0.5).astype(np.uint8)

                h0, w0 = original_sizes[i]
                res['planarity_probability'] = planarity
                res['planarity_probability_full'] = cv2.resize(planarity, (w0, h0))
                res['planarity_binary'] = planarity_bin
                res['planarity_binary_full'] = cv2.resize(
                    planarity_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)

                if return_all_heads:
                    if 'normal' in outputs:
                        res['normal'] = outputs['normal'][i].cpu().numpy()
                    if 'points' in outputs:
                        res['points'] = outputs['points'][i].cpu().numpy()

                results.append(res)
            return results


class MoGePlanarityProjNeckHeadInference:
    """Inference for proj_neck_head architecture: frozen backbone -> separate projections + neck + head."""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load frozen base MoGe model
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Deep-copy projections, neck, and normal_head for planarity
        self.planarity_projections = copy.deepcopy(self.model.encoder.output_projections).to(self.device)
        self.planarity_neck = copy.deepcopy(self.model.neck).to(self.device)
        self.planarity_head = copy.deepcopy(self.model.normal_head).to(self.device)
        _modify_head_last_layer(self.planarity_head, self.device)

        # Load trained weights
        self.planarity_projections.load_state_dict(checkpoint['planarity_projections_state_dict'])
        self.planarity_neck.load_state_dict(checkpoint['planarity_neck_state_dict'])
        self.planarity_head.load_state_dict(checkpoint['planarity_head_state_dict'])
        self.planarity_projections.eval()
        self.planarity_neck.eval()
        self.planarity_head.eval()

        # Cast to float32
        self.model = self.model.float().to(self.device)
        self.planarity_projections = self.planarity_projections.float().to(self.device)
        self.planarity_neck = self.planarity_neck.float().to(self.device)
        self.planarity_head = self.planarity_head.float().to(self.device)

        print("Model loaded successfully!")
        if 'epoch' in checkpoint:
            print(f"Model trained for {checkpoint['epoch']} epochs")
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")

    def _forward(self, images, num_tokens=1024):
        B, _, img_h, img_w = images.shape
        base_h, base_w = _compute_base_hw(images, num_tokens)

        # Frozen backbone only (raw intermediate layers, NOT full encoder.forward)
        with torch.no_grad():
            image_14 = F.interpolate(images, (base_h * 14, base_w * 14),
                                     mode='bilinear', align_corners=False, antialias=True)
            image_14 = (image_14 - self.model.encoder.image_mean) / self.model.encoder.image_std
            raw_layers = self.model.encoder.backbone.get_intermediate_layers(
                image_14, n=self.model.encoder.intermediate_layers, return_class_token=True
            )

        # Planarity: trainable projections -> sum -> neck -> head
        encoder_features = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (base_h, base_w)).contiguous())
            for proj, (feat, _) in zip(self.planarity_projections, raw_layers)
        ], dim=1).sum(dim=1)

        features_list = _build_features_list(encoder_features, images, base_h, base_w)
        planarity_neck_out = self.planarity_neck(features_list)
        planarity_logits = self.planarity_head(planarity_neck_out)[-1]
        planarity_logits = F.interpolate(planarity_logits, (img_h, img_w),
                                         mode='bilinear', align_corners=False)

        outputs = {
            'planarity_logits': planarity_logits,
            'planarity': torch.sigmoid(planarity_logits),
        }

        # Other heads: frozen model's original projections -> neck -> heads
        with torch.no_grad():
            frozen_encoder_features = torch.stack([
                proj(feat.permute(0, 2, 1).unflatten(2, (base_h, base_w)).contiguous())
                for proj, (feat, _) in zip(self.model.encoder.output_projections, raw_layers)
            ], dim=1).sum(dim=1)
            frozen_features_list = _build_features_list(
                frozen_encoder_features, images, base_h, base_w)
            frozen_neck_out = self.model.neck(frozen_features_list)

            # Apply same post-processing as MoGeModel.forward()
            if hasattr(self.model, 'points_head'):
                points_raw = self.model.points_head(frozen_neck_out)[-1]
                points_raw = F.interpolate(points_raw, (img_h, img_w),
                                           mode='bilinear', align_corners=False)
                points_raw = points_raw.permute(0, 2, 3, 1)
                outputs['points'] = self.model._remap_points(points_raw)

            if hasattr(self.model, 'normal_head'):
                normal_raw = self.model.normal_head(frozen_neck_out)[-1]
                normal_raw = F.interpolate(normal_raw, (img_h, img_w),
                                           mode='bilinear', align_corners=False)
                normal_raw = normal_raw.permute(0, 2, 3, 1)
                outputs['normal'] = F.normalize(normal_raw, dim=-1)

            if hasattr(self.model, 'mask_head'):
                mask_raw = self.model.mask_head(frozen_neck_out)[-1]
                mask_raw = F.interpolate(mask_raw, (img_h, img_w),
                                         mode='bilinear', align_corners=False)
                outputs['mask'] = mask_raw.squeeze(1).sigmoid()

        return outputs

    def preprocess_image(self, image_path, target_height=512, target_width=768):
        if isinstance(image_path, str):
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        original_h, original_w = image.shape[:2]
        image_resized = cv2.resize(image, (target_width, target_height))
        image_tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor, (original_h, original_w)

    def preprocess_images(self, image_paths, target_height=512, target_width=768):
        images = []
        original_sizes = []
        for p in image_paths:
            if isinstance(p, str):
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            else:
                img = p
            h, w = img.shape[:2]
            original_sizes.append((h, w))
            img_resized = cv2.resize(img, (target_width, target_height))
            img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
            images.append(img_tensor)
        images = torch.stack(images, dim=0).to(self.device)
        return images, original_sizes

    def predict(self, image_path, num_tokens=1024, return_all_heads=False):
        with torch.no_grad():
            image_tensor, original_size = self.preprocess_image(image_path)
            outputs = self._forward(image_tensor, num_tokens)

            results = {}
            if 'planarity' in outputs:
                planarity_prob = outputs['planarity'].squeeze().cpu().numpy()
                planarity_binary = (planarity_prob > 0.5).astype(np.uint8)
                results['planarity_probability'] = planarity_prob
                results['planarity_binary'] = planarity_binary

                if original_size != planarity_prob.shape:
                    results['planarity_probability_full'] = cv2.resize(
                        planarity_prob, (original_size[1], original_size[0]))
                    results['planarity_binary_full'] = cv2.resize(
                        planarity_binary, (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_NEAREST)

            if return_all_heads:
                if 'mask' in outputs:
                    results['mask'] = outputs['mask'].squeeze().cpu().numpy()
                if 'normal' in outputs:
                    # MoGe v2 format: (B, H, W, 3), squeeze gives (H, W, 3)
                    results['normal'] = outputs['normal'].squeeze(0).cpu().numpy()
                if 'points' in outputs:
                    # MoGe v2 format: (B, H, W, 3), squeeze gives (H, W, 3)
                    results['points'] = outputs['points'].squeeze(0).cpu().numpy()

            return results

    def predict_batch_fast(self, image_paths, num_tokens=1024, return_all_heads=False):
        with torch.no_grad():
            images, original_sizes = self.preprocess_images(image_paths)
            outputs = self._forward(images, num_tokens)

            B = images.shape[0]
            results = []
            for i in range(B):
                res = {}
                planarity = outputs['planarity'][i].squeeze().cpu().numpy()
                planarity_bin = (planarity > 0.5).astype(np.uint8)

                h0, w0 = original_sizes[i]
                res['planarity_probability'] = planarity
                res['planarity_probability_full'] = cv2.resize(planarity, (w0, h0))
                res['planarity_binary'] = planarity_bin
                res['planarity_binary_full'] = cv2.resize(
                    planarity_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)

                if return_all_heads:
                    if 'normal' in outputs:
                        res['normal'] = outputs['normal'][i].cpu().numpy()
                    if 'points' in outputs:
                        res['points'] = outputs['points'][i].cpu().numpy()

                results.append(res)
            return results
