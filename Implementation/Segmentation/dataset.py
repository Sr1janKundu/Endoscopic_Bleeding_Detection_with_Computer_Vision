import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BleedDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.mask_paths = []
        
        # Collect image and mask paths for each class
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            image_path = os.path.join(class_path, 'images')
            mask_path = os.path.join(class_path, 'annotations')

            image_files = sorted(os.listdir(image_path))
            mask_files = sorted(os.listdir(mask_path))

            # Ensure the number of images and masks match
            assert len(image_files) == len(mask_files), f"Mismatch in number of images and masks for class {class_name}"

            # Store paths
            self.image_paths.extend([os.path.join(image_path, img) for img in image_files])
            self.mask_paths.extend([os.path.join(mask_path, mask) for mask in mask_files])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)  # Convert to grayscale

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        # Convert to PyTorch tensors
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0   # Add a channel dim, then Normalize to [0, 1]

        return img, mask