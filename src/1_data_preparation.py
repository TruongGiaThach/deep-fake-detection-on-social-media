import os
import cv2
import threading
from queue import Queue
import pandas as pd
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from config import DATA_PROCESSED_CHECKPOINT_REAL, DATA_PROCESSED_CHECKPOINT_FAKE, FRAME_SAVE_PATH, LABEL_FILE, MIN_FACE_HEIGHT, MIN_FACE_WIDTH, MODEL_PATH, PROTO_PATH, REAL_PATH, FAKE_PATH, OUTPUT_FRAME_SIZE, FRAME_COUNT
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure frame save directories exist
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
os.makedirs(os.path.join(FRAME_SAVE_PATH, "real"), exist_ok=True)
os.makedirs(os.path.join(FRAME_SAVE_PATH, "fake"), exist_ok=True)

# Initialize label storage
labels_data = []

# Heavy full-quality face detector
detector = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
detector.prepare(ctx_id=0, det_size=(640, 640))

dnn_face_net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

def dnn_detect_faces(frame, conf_threshold=0.6):
    """
    Detect faces in a frame using OpenCV DNN.
    Returns a list of confidence scores.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))
    dnn_face_net.setInput(blob)
    detections = dnn_face_net.forward()
    confidences = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            confidences.append(confidence)
    return confidences

def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def feature_score(face):
    try:
        kps = face.kps  # shape (5, 2)
        if kps is None or len(kps) != 5:
            return 0

        # Check relative positions of keypoints
        left_eye, right_eye, nose, left_mouth, right_mouth = kps

        # Eyes should be horizontally aligned
        eye_diff = abs(left_eye[1] - right_eye[1])
        eye_dist = np.linalg.norm(left_eye - right_eye)

        # Nose should be roughly centered between eyes
        mid_eye = (left_eye + right_eye) / 2
        nose_offset = np.linalg.norm(nose - mid_eye)

        # Mouth corners should be below the eyes
        mouth_avg_y = (left_mouth[1] + right_mouth[1]) / 2
        eye_avg_y = (left_eye[1] + right_eye[1]) / 2
        mouth_below_eye = mouth_avg_y > eye_avg_y

        # Heuristic thresholds
        if eye_dist < 10:
            return 0  # eyes too close = probably bad detection
        if eye_diff > eye_dist * 0.25:
            return 1  # tilted or weird detection
        if nose_offset > eye_dist * 0.5:
            return 2  # off-center nose
        if not mouth_below_eye:
            return 3  # invalid mouth position

        return 5  # All looks normal
    except:
        return 0

def score_face(face):
    yaw, pitch = abs(face.pose[0]), abs(face.pose[1])

    # Frontal score: negative sum of yaw and pitch (higher is better for frontal)
    frontal_score = -(yaw + pitch)
    # Size score: prioritize larger faces
    width = face.bbox[2] - face.bbox[0]
    height = face.bbox[3] - face.bbox[1]
    size_score = (width * height)  # Face area

    # if width < MIN_FACE_HEIGHT or height < MIN_FACE_WIDTH:  # example threshold
    #     size_score=size_score/2
    # get face features score
    feature = feature_score(face)

    # Increase the weight on size and penalize on face that is not head-on
    return {
        'face': face,
        'size': size_score,
        'frontal': frontal_score,
        'feature': feature,
    }

def normalize_scores(scored_faces):
    size_scores = np.array([s['size'] for s in scored_faces], dtype=np.float32)
    frontal_scores = np.array([s['frontal'] for s in scored_faces], dtype=np.float32)
    feature_scores = np.array([s['feature'] for s in scored_faces], dtype=np.float32)

    def normalize(arr):
        arr = np.array(arr, dtype=np.float32)
        max_val = arr.max()
        if max_val < 1e-5:
            return np.full_like(arr, 50.0)  # fallback if all values are zero or near zero
        return (arr / max_val) * 50.0

    norm_size = normalize(size_scores)

    for i in range(len(scored_faces)):
        scored_faces[i]['norm_score'] = norm_size[i] + frontal_scores[i] + feature_scores[i] * 2
    return scored_faces

def safe_crop(face, frame_rgb):
    # Clamping Coordinates
    h, w, _ = frame_rgb.shape
    x1 = max(0, min(w, int(face.bbox[0])))
    y1 = max(0, min(h, int(face.bbox[1])))
    x2 = max(0, min(w, int(face.bbox[2])))
    y2 = max(0, min(h, int(face.bbox[3])))

    # Invalid Box Check
    if x2 <= x1 or y2 <= y1:
        return None

    return frame_rgb[y1:y2, x1:x2]

# ================================
# === Batched Detection Function
# ================================

def detect_and_crop_best_face(frame_rgb):
    faces = detector.get(frame_rgb)
    
    if not faces:
        return None

    # Filter and prioritize faces based on new scoring system
    scored_faces = [score_face(face) for face in faces]

    if not scored_faces:
        return None

    # Get the face with the highest score
    scored_faces = normalize_scores(scored_faces)
    
    # Sort faces by normalized score in descending order
    sorted_faces = sorted(scored_faces, key=lambda x: x['norm_score'], reverse=True)

    # First pass: try to find a non-blurry face
    for scored in sorted_faces:
        best_face = scored['face']
        if best_face:
            face_crop = safe_crop(best_face, frame_rgb)
            if face_crop.size == 0:
                continue  # Crop failed, try next

            # Blur filter check
            if blur_score(face_crop) < 8:  # You can tune this threshold
                continue  # Too blurry, try next

            return cv2.resize(face_crop, OUTPUT_FRAME_SIZE)
        
    # Second pass: fallback to best valid face without blur check
    for scored in sorted_faces:
        best_face = scored['face']
        if best_face:
            face_crop = safe_crop(best_face, frame_rgb)
            if face_crop.size == 0:
                continue
            return cv2.resize(face_crop, OUTPUT_FRAME_SIZE)

    # No valid face found
    return None

# Function to extract frames, detect faces, and save as images
def extract_and_save_frames(video_path, tag, save_dir, label, max_frames=10, writer=None):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    candidate_frames = []  # List of tuples: (frame_index, confidence, full_resolution_frame)

    candidate_target = 3 * max_frames
    frame_gap = max(1, total_frames // (candidate_target + 1))
    
    batch_samples = []     # Low-res samples for DNN detection (300x300)
    batch_indices = []     # Corresponding frame indices
    batch_full_frames = [] # Full-resolution copies for later processing

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_gap == 0:
            small_frame = cv2.resize(frame, (300, 300))
            batch_samples.append(small_frame)
            batch_indices.append(frame_idx)
            batch_full_frames.append(frame.copy())

            if len(batch_samples) >= 8:
                for i, sample in enumerate(batch_samples):
                    detections = dnn_detect_faces(sample)
                    if detections:
                        candidate_frames.append((batch_indices[i], max(detections), batch_full_frames[i]))
                batch_samples.clear()
                batch_indices.clear()
                batch_full_frames.clear()

                if len(candidate_frames) >= candidate_target:
                    break
        frame_idx += 1

    if batch_samples:
        for i, sample in enumerate(batch_samples):
            detections = dnn_detect_faces(sample)
            if detections:
                candidate_frames.append((batch_indices[i], max(detections), batch_full_frames[i]))

    candidate_frames = sorted(candidate_frames, key=lambda x: -x[1])
    saved_count = 0
    for idx, conf, full_frame in candidate_frames:
        frame_rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
        best_face = detect_and_crop_best_face(frame_rgb)
        if best_face is not None:
            frame_filename = f"{tag}_{video_name}_frame_{saved_count}.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            if writer:
                writer.submit(cv2.imwrite, frame_path, cv2.cvtColor(best_face, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(frame_path, cv2.cvtColor(best_face, cv2.COLOR_RGB2BGR))
            labels_data.append([frame_filename, label])
            saved_count += 1
        if saved_count >= max_frames:
            break

    cap.release()
    if saved_count == 0:
        print(f"No usable faces found in {video_path}")

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()

def save_checkpoint(file_path, folder_name):
    with open(file_path, "a") as f:
        f.write(folder_name + "\n")

def save_labels():
    if labels_data:
        if os.path.exists(LABEL_FILE):
            df_existing = pd.read_csv(LABEL_FILE)
            df_new = pd.DataFrame(labels_data, columns=["filename", "label"])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame(labels_data, columns=["filename", "label"])
        df_combined.to_csv(LABEL_FILE, index=False)

# Writer Thread
class AsyncWriter(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = Queue()
        self.running = True

    def run(self):
        while self.running or not self.queue.empty():
            task = self.queue.get()
            if task is None:
                break
            func, args = task
            try:
                func(*args)
            except Exception as e:
                print(f"[Writer Error] {e}")
            self.queue.task_done()

    def submit(self, func, *args):
        self.queue.put((func, args))

    def stop(self):
        self.running = False
        self.queue.put(None)

def main():
    global labels_data
    writer = AsyncWriter()
    writer.start()

    # # Process and save real video frames
    processed_real = load_checkpoint(DATA_PROCESSED_CHECKPOINT_REAL)
    real_subfolders = [subfolder for subfolder in os.listdir(REAL_PATH) if os.path.isdir(os.path.join(REAL_PATH, subfolder))]

    for real_folder in real_subfolders:
        if real_folder in processed_real:
            # Skipping already processed real folder
            continue

        frames_path = os.path.join(FRAME_SAVE_PATH, "real", real_folder)
        os.makedirs(frames_path, exist_ok=True)
        folder_path = os.path.join(REAL_PATH, real_folder)
        print(f"Processing real videos from {folder_path}...")
        real_videos = os.listdir(folder_path)

        for video_file in tqdm(real_videos):
            video_path = os.path.join(folder_path, video_file)
            extract_and_save_frames(video_path, real_folder, frames_path, 0, FRAME_COUNT, writer)
            
        # Save progress
        save_checkpoint(DATA_PROCESSED_CHECKPOINT_REAL, real_folder)
        save_labels()
        labels_data = []  # Clear after saving

    # Process fake videos from multiple subfolders
    processed_fake = load_checkpoint(DATA_PROCESSED_CHECKPOINT_FAKE)
    fake_subfolders = [subfolder for subfolder in os.listdir(FAKE_PATH) if os.path.isdir(os.path.join(FAKE_PATH, subfolder))]

    for fake_folder in fake_subfolders:
        if fake_folder in processed_fake:
            # Skipping already processed real folder
            continue

        frames_path = os.path.join(FRAME_SAVE_PATH, "fake", fake_folder)
        os.makedirs(frames_path, exist_ok=True)
        folder_path = os.path.join(FAKE_PATH, fake_folder)
        print(f"Processing fake videos from {folder_path}...")
        fake_videos = os.listdir(folder_path)
        for video_file in tqdm(fake_videos):
            video_path = os.path.join(folder_path, video_file)
            extract_and_save_frames(video_path, fake_folder, frames_path, 1, FRAME_COUNT, writer)

        # Save progress
        save_checkpoint(DATA_PROCESSED_CHECKPOINT_FAKE, fake_folder)
        save_labels()
        labels_data = []  # Clear after saving

    writer.stop()
    writer.join()

def test():
    # Process and save real video frames
    writer = AsyncWriter()
    writer.start()
    real_folder = "original"

    frames_path = os.path.join(FRAME_SAVE_PATH, "real", real_folder)
    os.makedirs(frames_path, exist_ok=True)
    folder_path = os.path.join(REAL_PATH, real_folder)

    fake_videos = ["014.mp4","060.mp4","091.mp4","156.mp4", "190.mp4", "344.mp4", "508.mp4", "581.mp4", "607.mp4"]
    for video_file in tqdm(fake_videos):
        video_path = os.path.join(folder_path, video_file)
        extract_and_save_frames(video_path, real_folder, frames_path, 0, FRAME_COUNT, writer)
    writer.stop()
    writer.join()

if __name__ == '__main__':
    main()
