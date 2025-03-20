import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from efficient_net_lstm import EfficientNetLSTM
from config import FRAME_SAVE_PATH, LABEL_FILE

# Configuration
MODEL_PATH = "check_points/bestval.pth"  # Change to your model path
IMAGE_FOLDER = "data/frames/fake"  # Change to your folder containing images
RESULT_CSV = "detection_result.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Dataset class for loading images
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

# Load the model
def load_model(model_path):
    model = EfficientNetLSTM().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def batch_forward(model: EfficientNetLSTM, device: torch.device, data, labels):
    data = data.to(device)
    labels = labels.to(device)
    out = model(data)
    _, pred = torch.max(out, 1)
    return pred


# Detect deepfakes
def detect_deepfakes(model, dataloader):
    count_0, count_1 = 0, 0
    count_real, count_deepfake = 0, 0
    total = correct= 0
    with torch.no_grad():
        for images, filenames in dataloader:
            images, filenames = images.to(device), filenames.to(device)
            
            preds = batch_forward(model, device, images, filenames)
            total += filenames.size(0)
            correct += (preds == filenames).sum().item()
            for filename, pred in zip(filenames, preds.cpu().numpy()):
                if filename == 0:
                    count_0 += 1
                elif filename == 1:
                    count_1 += 1
                
                if pred == 1:
                    count_deepfake += 1
                else:
                    count_real += 1
   
       
    print(f"Số ảnh riu: {count_0}")
    print(f"Số ảnh phake: {count_1}")
    print(f"Pha kè: {count_deepfake}")
    print(f"Rì eo: {count_real}")

    train_acc = correct / total

    print(f"Accuracy: {train_acc}")

if __name__ == "__main__":
    dataset = DeepfakeDataset(LABEL_FILE, FRAME_SAVE_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    model = load_model(MODEL_PATH)
    detect_deepfakes(model, dataloader)

