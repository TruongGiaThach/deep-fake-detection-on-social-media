import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Iterable
from PIL import Image
from config import OUTPUT_FRAME_SIZE
from torchvision import transforms
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load images from CSV
def load_images_from_folder(folder, label):
    data = []
    labels = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, OUTPUT_FRAME_SIZE)
            data.append(img)
            labels.append(label)
    return np.array(data), np.array(labels)

# Data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

def augment_frames(frames):
    return np.array([datagen.random_transform(frame) for frame in frames])


# def get_transformer(size: int): 
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),  # Flip images randomly
#         transforms.RandomRotation(10),  # Rotate slightly
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color adjustments
#         transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
#         transforms.Resize((size, size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     return transform
