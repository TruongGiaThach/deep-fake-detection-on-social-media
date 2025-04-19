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

        # Làm sạch cột tên file (cột đầu tiên) để chỉ chứa tên file
        self.labels_df.iloc[:, 0] = self.labels_df.iloc[:, 0].apply(
            lambda x: os.path.basename(x.replace('\\', '/')).strip()
        )

        # Shuffle dataset to ensure mixing of real and fake
        self.labels_df = self.labels_df.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.labels_df) // self.seq_len  # Each sample is a sequence
    
    def __getitem__(self, idx):
        images = []
        seq_indices = np.random.choice(len(self.labels_df), self.seq_len, replace=False)  # Randomize frames

        for frame_idx in seq_indices:
            # Lấy tên file và đảm bảo làm sạch
            img_filename = os.path.basename(self.labels_df.iloc[frame_idx, 0].replace('\\', '/')).strip()
            
            # Tạo đường dẫn
            img_name = os.path.join(
                self.root_dir, 
                "real" if self.labels_df.iloc[frame_idx, 1] == 0 else "fake", 
                img_filename
            )
            # Chuẩn hóa đường dẫn
            img_name = os.path.normpath(img_name)
            
            # Kiểm tra file tồn tại
            if not os.path.exists(img_name):
                raise FileNotFoundError(
                    f"File not found: {img_name}\n"
                    f"Raw filename: {self.labels_df.iloc[frame_idx, 0]}\n"
                    f"Root dir: {self.root_dir}"
                )
            
            image = Image.open(img_name).convert('RGB')

            if self.transform:
                image = self.transform(image)

            images.append(image)

        images = torch.stack(images)  # Shape: (seq_len, C, H, W)
        label = self.labels_df.iloc[seq_indices[0], 1]  # Label from first frame in sequence

        return images, label