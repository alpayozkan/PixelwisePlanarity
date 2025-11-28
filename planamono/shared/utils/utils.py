import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
from tqdm import tqdm

from plyfile import PlyData
import numpy as np

from scipy import stats

import imageio

def save_label_image_sem(path, label_img):
    label_img = np.asarray(label_img)
    dtype = np.uint16
    
    max_val = label_img.max()
    if abs(max_val) > 65535:
        raise ValueError(f"Label value too large: {max_val}, use .npy instead")
    # Replace -1 (optional) to reserved 0
    label_img += 1 # unannotated -100, -1 no raycast, 0 wall =>  -99, 0, 1, ...
    label_img[label_img < 0] = 0
    imageio.imwrite(path, label_img.astype(dtype))

def save_label_image(path, label_img):
    label_img = np.asarray(label_img)

    # Decide dtype based on max value
    max_val = label_img.max()
    if max_val <= 255:
        dtype = np.uint8
    elif max_val <= 65535:
        dtype = np.uint16
    else:
        raise ValueError(f"Label value too large: {max_val}, use .npy instead")
    # Replace -1 (optional) to reserved 0
    label_img[label_img == -1] = 0

    imageio.imwrite(path, label_img.astype(dtype))


def load_label_image(path):
    label_img = imageio.imread(path)
    # Convert to int32 for downstream consistency
    return label_img.astype(np.int32)

def remap_semantic(semantic_img):
    unique_labels = np.unique(semantic_img)
    # print("Original labels:", unique_labels)
    label_remap_dict = {old_label: new_index for new_index, old_label in enumerate(unique_labels)}
    remapped_semantic_img = np.vectorize(label_remap_dict.get)(semantic_img)
    return remapped_semantic_img


def load_semantic_id_to_name_list(txt_path):
    """
    Loads semantic_classes.txt where each line is a class name.
    Returns a dict: {id: class_name}, with line number as id.
    """
    id_to_name = {}
    with open(txt_path, "r") as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                id_to_name[idx] = class_name
    return id_to_name

def labels_to_names(label_array, id_to_name):
    # Vectorized map using np.vectorize
    map_func = np.vectorize(lambda x: id_to_name.get(x, "unknown"))
    return map_func(label_array)




def get_random_cmap(num_classes, seed=42):
    import numpy as np
    np.random.seed(seed)
    colors = np.random.rand(num_classes, 3)
    return mcolors.ListedColormap(colors)

