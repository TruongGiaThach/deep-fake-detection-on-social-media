import os
import cv2
import pandas as pd
from tqdm import tqdm
import torch
from mtcnn import MTCNN
from config import FRAME_SAVE_PATH, LABEL_FILE, REAL_PATH, FAKE_PATH, OUTPUT_FRAME_SIZE, FRAME_COUNT, MAX_VIDEOS

# Ensure frame save directories exist
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(FRAME_SAVE_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(FRAME_SAVE_PATH, "fake"), exist_ok=True)

# Initialize label storage
labels_data = []
detector = MTCNN()  # Initialize face detector

# Function to extract frames, detect faces, and save as images with GPU acceleration
def extract_and_save_frames(video_path, save_dir, label, frame_count=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i} from {video_path}")
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_boxes = detector.detect_faces(frame_rgb)
        
        if face_boxes and len(face_boxes) > 0:
            x, y, w, h = face_boxes[0]['box']
            face_crop = frame_rgb[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, OUTPUT_FRAME_SIZE)
        else:
            face_resized = cv2.resize(frame_rgb, OUTPUT_FRAME_SIZE)
        
        frame_filename = f"{video_name}_frame{i}.jpg"
        frame_path = os.path.join(save_dir, frame_filename)
        cv2.imwrite(frame_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
        labels_data.append([frame_filename, label])
    cap.release()

# Process and save real video frames
print("Processing and saving real video frames...")
real_videos = os.listdir(REAL_PATH)[:MAX_VIDEOS]
for video_file in tqdm(real_videos):
    video_path = os.path.join(REAL_PATH, video_file)
    extract_and_save_frames(video_path, os.path.join(FRAME_SAVE_PATH, "real"), 0)

# Process and save fake video frames
print("Processing and saving fake video frames...")
fake_videos = os.listdir(FAKE_PATH)[:MAX_VIDEOS]
for video_file in tqdm(fake_videos):
    video_path = os.path.join(FAKE_PATH, video_file)
    extract_and_save_frames(video_path, os.path.join(FRAME_SAVE_PATH, "fake"), 1)

# Save labels to CSV and shuffle data
labels_df = pd.DataFrame(labels_data, columns=["filename", "label"])
labels_df = labels_df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset
labels_df.to_csv(LABEL_FILE, index=False)