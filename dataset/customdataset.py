import os
import random

import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy import linalg

class CustomDataset(Dataset):  # Change Dataset to CustomDataset
    rgb_from_her = np.array([[0.65, 0.70, 0.29], # H
                            [0.07, 0.99, 0.11],  # E
                            [0.00, 0.00, 0.00]]) # R
    rgb_from_her[2, :] = np.cross(rgb_from_her[0, :], rgb_from_her[1, :])
    her_from_rgb = linalg.inv(rgb_from_her)

    def __init__(self, dataset_dir, image_transform, target_transform, mode, her_image=False):
        self.dataset_dir = dataset_dir
        self.samples_paths = glob("{}/*.npy".format(os.path.join(self.dataset_dir, mode)))  # Fix the glob pattern

        self.image_transform = image_transform
        self.target_transform = target_transform
        self.mode = mode
        self.her_image = her_image

    def __len__(self):
        return len(self.samples_paths)

    def __getitem__(self, idx):
        sample = np.load(self.samples_paths[idx])

        image = (sample[:, :, :3] * 255.0).astype(np.uint8)
        nuclear_mask = (sample[:, :, 4]).astype(np.float32)

        if self.her_image:
            image_blur = cv2.GaussianBlur(image, (5, 5), 0)
            her_image = self.deconv_stains(image_blur, self.her_from_rgb)

        if self.image_transform:
            seed = random.randint(0, 2 ** 32)
            self._set_seed(seed); image = self.image_transform(image)
            self._set_seed(seed); nuclear_mask = self.target_transform(nuclear_mask)
            if self.her_image:
                self._set_seed(seed); her_image = self.image_transform(her_image)

        nuclear_mask[nuclear_mask >= 0.5] = 1; nuclear_mask[nuclear_mask < 0.5] = 0

        if self.her_image:
            return image, nuclear_mask, her_image
        else:
            return image, nuclear_mask

    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def deconv_stains(self, rgb, conv_matrix):
        # change datatype to float64
        rgb = (rgb).astype(np.float64)
        np.maximum(rgb, 1E-6,
                   out=rgb)  # to avoid log artifacts <- 로그 함수는 입력값이 0 에 가까워질수록, 음의 무한대로 수렴, 그래서 0 에 근접하면 아티팩트가 발생할 수 있음
        log_adjust = np.log(1E-6)  # for compensate the sum above
        x = np.log(rgb)
        stains = (x / log_adjust) @ conv_matrix

        # normalizing and shifting the data distribution to proper pixel values range (i.e., [0,255])
        h = 1 - (stains[:, :, 0] - np.min(stains[:, :, 0])) / (np.max(stains[:, :, 0]) - np.min(stains[:, :, 0]))
        e = 1 - (stains[:, :, 1] - np.min(stains[:, :, 1])) / (np.max(stains[:, :, 1]) - np.min(stains[:, :, 1]))
        r = 1 - (stains[:, :, 2] - np.min(stains[:, :, 2])) / (np.max(stains[:, :, 2]) - np.min(stains[:, :, 2]))

        her = cv2.merge((h, e, r)) * 255

        return her.astype(np.uint8)