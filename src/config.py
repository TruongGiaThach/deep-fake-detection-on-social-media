# config.py

# Dataset paths
DATASET_PATH = "dataset"
REAL_PATH = "dataset/videos/real"
FAKE_PATH = "dataset/videos/fake"

# Frame extraction settings
OUTPUT_FRAME_SIZE = (224, 224)
FACE_SIZE = 224
MIN_FACE_HEIGHT=40
MIN_FACE_WIDTH=40
FRAME_COUNT = 10  
BATCH_SIZE = 32

# Data paths
FRAME_SAVE_PATH = "dataset/frames/"
LABEL_FILE = "dataset/frames/labels.csv"
DATA_PROCESSED_CHECKPOINT_REAL = "data/frames/real_checkpoints.txt"
DATA_PROCESSED_CHECKPOINT_FAKE = "data/frames/fake_checkpoints.txt"

# Model paths
PROTO_PATH = "models/dnn_face/deploy.prototxt"
MODEL_PATH = "models/dnn_face/res10_300x300_ssd_iter_140000.caffemodel"
MODEL_SAVE_PATH = "models/result/deepfake_detection_model.pth"
MODEL_CHECKPOINT = "check_points"

# LogS paths
LOGS_PATH = "logs"