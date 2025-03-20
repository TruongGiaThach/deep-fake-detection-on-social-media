# config.py

# Dataset paths
REAL_PATH = "../dataset/DFD_original_sequences"
FAKE_PATH = "../dataset/DFD_manipulated_sequences/DFD_manipulated_sequences"

# Frame extraction settings
OUTPUT_FRAME_SIZE = (128, 128)
FACE_SIZE = 128
FRAME_COUNT = 10  
MAX_VIDEOS = 700  

# Data paths
FRAME_SAVE_PATH = "data/frames/"
LABEL_FILE = "data/frames/labels.csv"

# Model paths
MODEL_SAVE_PATH = "models/deepfake_detection_model.pth"
CHECKPOINT_SAVE_PATH = "check_points"

# LogS paths
LOGS_PATH = "logs"