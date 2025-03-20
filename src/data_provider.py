import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Process Dataset to different set of data
class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, seq_len=10):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len

        # Shuffle dataset to ensure mixing of real and fake
        self.labels_df = self.labels_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df) // self.seq_len  # Each sample is a sequence
    
    def __getitem__(self, idx):
        images = []
        seq_indices = np.random.choice(len(self.labels_df), self.seq_len, replace=False)  # Randomize frames

        for frame_idx in seq_indices:
            img_name = os.path.join(
                self.root_dir, 
                "real" if self.labels_df.iloc[frame_idx, 1] == 0 else "fake", 
                self.labels_df.iloc[frame_idx, 0]
            )
            image = Image.open(img_name).convert('RGB')

            if self.transform:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images)  # Shape: (seq_len, C, H, W)
        label = self.labels_df.iloc[seq_indices[0], 1]  # Label from first frame in sequence

        return images, label