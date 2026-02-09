from torch.utils.data import Dataset
import random

import torchvision.transforms.functional as TF

class MixedPlanarityDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, image_height=476, image_width=644, seed=42):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        self.total_len = self.len_a + self.len_b

        self.prob_a = self.len_a / self.total_len
        self.image_height = image_height
        self.image_width = image_width
        random.seed(seed)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        sample = self.dataset_a[idx % self.len_a] if random.random() < self.prob_a else self.dataset_b[idx % self.len_b]

        # Resize tensors
        H, W = self.image_height, self.image_width
        sample["image"] = TF.resize(sample["image"], [H, W], interpolation=TF.InterpolationMode.BILINEAR)
        sample["depth"] = TF.resize(sample["depth"], [H, W], interpolation=TF.InterpolationMode.BILINEAR)
        sample["plane"] = TF.resize(sample["plane"], [H, W], interpolation=TF.InterpolationMode.NEAREST)
        sample["semantic"] = TF.resize(sample["semantic"], [H, W], interpolation=TF.InterpolationMode.NEAREST)

        return sample
