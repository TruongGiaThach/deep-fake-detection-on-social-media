import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import FRAME_SAVE_PATH, LABEL_FILE, OUTPUT_FRAME_SIZE, FRAME_COUNT
import random

# Threshold for blur detection
# BLUR_THRESHOLD = 100.0

# def is_blurry(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return variance < BLUR_THRESHOLD

def apply_random_augmentation(image):
    img = image.copy()
    h, w = img.shape[:2]  # Moved here to ensure w and h are always defined

    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    
    # Random brightness and contrast adjustment
    if random.random() < 0.5:
        alpha = 1.0 + (0.6 * (np.random.rand() - 0.5))  # contrast factor between ~0.7 to 1.3
        beta = 50 * (np.random.rand() - 0.5)            # brightness shift between -25 and +25
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Random rotation with larger angle
    if random.random() < 0.5:
        angle = np.random.uniform(-15, 15)  # rotate between -15 and 15 degrees
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)
    
    # Random scaling and translation
    if random.random() < 0.4:
        scale = np.random.uniform(0.9, 1.1)  # scale between 90% and 110%
        tx = np.random.uniform(-0.05 * w, 0.05 * w)  # translate x
        ty = np.random.uniform(-0.05 * h, 0.05 * h)  # translate y
        M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT101)

    return img


def load_images_grouped():
    df = pd.read_csv(LABEL_FILE)
    df['subfolder'] = df['filename'].apply(lambda x: x.split('_')[0])
    df['group'] = df['filename'].apply(lambda x: '_'.join(x.split('_')[:-2]))

    X_all = []
    y_all = []
    removed_indices = []
    group_names = []

    for group_name, group_df in tqdm(df.groupby('group')):
        group_imgs = []
        label = group_df.iloc[0].label
        subfolder = group_df.iloc[0].subfolder
        for idx, row in group_df.iterrows():
            path = os.path.join(FRAME_SAVE_PATH, "real" if row.label == 0 else "fake", row.filename)
            if not os.path.exists(path):
                removed_indices.append(idx)
                continue
            img = cv2.imread(path)
            # if is_blurry(img):
            #     os.remove(path)
            #     removed_indices.append(idx)
            #     continue
            img_resized = cv2.resize(img, OUTPUT_FRAME_SIZE)
            group_imgs.append(img_resized)

        if len(group_imgs) > 0:
            X_all.append((group_name, group_imgs, label, subfolder))
            y_all.append(label)
            group_names.append(group_name)

    df.drop(index=removed_indices, inplace=True)
    df.to_csv(LABEL_FILE, index=False)
    return X_all, y_all, df, group_names

def augment_group_with_transforms(group_name, group_imgs, label, needed_count, start_idx):
    new_labels = []
    for i in range(needed_count):
        chosen_img = random.choice(group_imgs)
        aug_img = apply_random_augmentation(chosen_img)
        fname = f"{group_name}_aug_{i+start_idx}.jpg"
        save_path = os.path.join(FRAME_SAVE_PATH, "real" if label == 0 else "fake", fname)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, aug_img)
        new_labels.append([fname, label])
    return new_labels

def main():
    print("Filtering blurry images and preparing for augmentation...")
    X_all, y_all, df, group_names = load_images_grouped()
    new_labels_total = []

    # 1. Fill missing frames per video
    for idx, (group_name, group_imgs, label, subfolder) in enumerate(X_all):
        missing_count = FRAME_COUNT - len(group_imgs)
        if missing_count <= 0:
            continue
        new_labels = augment_group_with_transforms(group_name, group_imgs, label, missing_count, 0)
        new_labels_total.extend(new_labels)

    # 2. Balance real vs fake
    real_count = len(df[df.label == 0]) + len([l for l in new_labels_total if l[1] == 0])
    fake_count = len(df[df.label == 1]) + len([l for l in new_labels_total if l[1] == 1])
    imbalance = fake_count - real_count

    if imbalance > 0:
        print(f"Balancing real samples: need to generate {imbalance} augmented real frames...")
        all_real_entries = df[df.label == 0]
        real_imgs_data = []

        for _, row in tqdm(all_real_entries.iterrows(), total=all_real_entries.shape[0]):
            path = os.path.join(FRAME_SAVE_PATH, "real", row.filename)
            if not os.path.exists(path):
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            img_resized = cv2.resize(img, OUTPUT_FRAME_SIZE)
            real_imgs_data.append(img_resized)

        for i in range(imbalance):
            chosen_img = random.choice(real_imgs_data)
            aug_img = apply_random_augmentation(chosen_img)
            fname = f"balanced_real_aug_{i}.jpg"
            save_path = os.path.join(FRAME_SAVE_PATH, "real", "balanced", fname)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, aug_img)
            new_labels_total.append([os.path.join("balanced", fname), 0])

    df_aug = pd.DataFrame(new_labels_total, columns=["filename", "label"])
    df_combined = pd.concat([df[["filename", "label"]], df_aug], ignore_index=True)
    df_combined.to_csv(LABEL_FILE, index=False)
    print("Frame augmentation and real/fake balancing complete.")

if __name__ == '__main__':
    main()
