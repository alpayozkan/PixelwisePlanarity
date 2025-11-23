from eval import segmentation_covering, evaluate_planarity
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np
import pandas as pd

import time

import sys
sys.path.append("/cluster/home/aoezkan/planeseg/3d_vision/planarity_2_segmentation")
import plan2seg
import utils
import postprocess
import visualize_seg

import time
import copy
import json
import imageio

import h5py
from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information

sys.path.append("/cluster/home/aoezkan/planeseg/3d_vision/homography")
from homog_utils import remap_labels
# from homog import *
# from homog_warp import *

sys.path.append("/cluster/home/aoezkan/planeseg/3d_vision/plane_fitting")
from planefit_visualize import *
from planefit import *
from planefit_utils import *
from planefit_metrics import *
from planeseg_visualize import *

sys.path.append("/cluster/home/aoezkan/planeseg/3d_vision/monocular/moge")
from inference import MoGePlanarityInference
import argparse
import torch
from PIL import Image

import torch
import torch.nn.functional as F

from natsort import natsorted
from tqdm import tqdm
import glob
from PIL import Image

import poselib
import os


import torch
import numpy as np
import random

sys.path.append('/cluster/home/aoezkan/planeseg/3d_vision/dataset/scannetpp')
from dataset_scannet_plane import ScanNetPPPlaneDataset  # <- replace with your actual import
from torch.utils.data import DataLoader






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate planarity metrics")

    # === General args ===
    parser.add_argument("--method", type=str, required=True, choices=["moge", "planercnn", "zeroplane", "gt", "monoplane"],
                        help="Which method to evaluate (moge, planercnn, zeroplane, gt)")
    parser.add_argument("--model_path", type=str,
                        help="Path to trained MoGe checkpoint (.pt) if method=moge")
    parser.add_argument("--model_size", type=str, default="middle",
                        choices=["small", "middle", "large"],
                        help="MoGe model size (if method=moge)")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--max_scenes", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--res_h", type=int, default=480)
    parser.add_argument("--res_w", type=int, default=640)

    args = parser.parse_args()


    dataset_dir = '/cluster/scratch/aoezkan/dataset/scannetpp'

    val_dataset = ScanNetPPPlaneDataset(
        rgb_root="/cluster/project/cvg/Shared_datasets/scannet++/data",
        plane_label_root=os.path.join(dataset_dir, "plane_ours_gt"),
        sem_label_root=os.path.join(dataset_dir, "semantic_gt"),
        depth_label_root=os.path.join(dataset_dir, "depth_gt_rendered"),
        split_txt_dir=os.path.join(dataset_dir, "splits"),
        split="val",
        max_scenes=args.max_scenes,
    )

    num_workers = 4
    batch_size = 1

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )


    if args.method == "moge":
        assert args.model_path is not None, "Please provide --model_path for MoGe"
        inference_model = MoGePlanarityInference(
            args.model_path, model_size='large', device=args.device
        )
        
        # self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        inference_model.model.encoder.use_memory_efficient_attention = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        inference_model.model = inference_model.model.half()
        inference_model.model.encoder.enable_pytorch_native_sdpa()

    elif args.method == "monoplane":
        assert args.model_path is not None, "Please provide --model_path for MoGe"
        inference_model = MoGePlanarityInference(
            args.model_path, model_size=args.model_size, device=args.device
        )
        inference_model.model.encoder.use_memory_efficient_attention = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        inference_model.model = inference_model.model.half()
        inference_model.model.encoder.enable_pytorch_native_sdpa()

    else:
        inference_model = None
        

    # save_dir = '/cluster/scratch/aoezkan/dataset/scannetpp/results/metrics'

    H,W = 480, 640
    # df_moge = evaluate_planarity(val_loader, inference_model, tag="moge", img_res=(H,W))
    # df_moge.to_csv(f"{save_dir}/eval_moge.csv", index=False)
    print('METHOD: ', args.method)

    os.makedirs(args.save_dir, exist_ok=True)
    df = evaluate_planarity(val_loader, inference_model, tag=args.method, img_res=(args.res_h, args.res_w))
    
    csv_path = os.path.join(args.save_dir, f"eval_{args.method}_{args.max_scenes}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] → {csv_path}")