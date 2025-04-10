# DeepFake Detection Model Training

This repository contains code for training a deep learning model to detect deepfake videos using a hybrid CNN-LSTM architecture.

## Project Description

This project implements a CNN-LSTM-based deepfake detection model designed to identify manipulated video content by analyzing both spatial and temporal features. The model leverages Convolutional Neural Networks (CNNs) to extract spatial features from video frames (e.g., facial textures, landmarks) and Long Short-Term Memory (LSTM) networks to analyze temporal patterns across frame sequences, detecting inconsistencies indicative of deepfake manipulation. The approach is inspired by state-of-the-art techniques reviewed in _An Investigation into the Utilisation of CNN with LSTM for Video Deepfake Detection_ (Tipper et al., 2024).

Key features:

-   Hybrid CNN-LSTM architecture for spatial-temporal analysis
-   Facial feature extraction using MTCNN and dlib for preprocessing
-   Training and evaluation on the FaceForensics++ dataset
-   Performance assessment using Accuracy, Precision, Recall, and F1-Score
-   Data augmentation to enhance model robustness

## Directory Structure

The repository is organized as follows:

```
deep-fake-detection-on-social-media/
│
├── dataset/
│   ├── raw/                # Raw FaceForensics++ dataset (real and manipulated videos)
│   ├── processed/          # Preprocessed frame sequences and facial features
│
├── models/
│   ├── checkpoints/        # Saved model checkpoints during training
│   ├── pretrained/         # Pretrained CNN models (e.g., ResNet, Xception) for transfer learning
│
├── scripts/
│   ├── train.py            # Script for training the CNN-LSTM model
│   ├── evaluate.py         # Script for evaluating model performance
│   ├── preprocess.py       # Script for video frame extraction and facial feature preprocessing
│
├── notebooks/
│   ├── exploration.ipynb   # Jupyter notebook for exploring the FaceForensics++ dataset
│   ├── visualization.ipynb # Notebook for visualizing detection results and metrics
│
├── results/
│   ├── logs/               # Training and evaluation logs
│   ├── metrics/            # Saved evaluation metrics (Accuracy, Precision, Recall, F1-Score)
│
└── README.md               # Project documentation
```

## Getting Started

To set up and run the project, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://gitlab.com/danghaithinh/deep-fake-detection-on-social-media.git
    cd deep-fake-detection-on-social-media
    ```

2. **Install dependencies**:

    Install the required Python packages, including TensorFlow/PyTorch, OpenCV, dlib, and others:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset**:

    Download the FaceForensics++ dataset from [Kaggle](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data) and preprocess it:

    ```bash
    python scripts/preprocess.py --input dataset/raw --output dataset/processed
    ```

    - The preprocessing script extracts faces using MTCNN, identifies facial landmarks with dlib (e.g., eyes, nose, mouth), and applies data augmentation techniques (flipping, rotation, brightness adjustment, Gaussian noise).

4. **Train the model**:

    Train the CNN-LSTM model using the preprocessed data:

    ```bash
    python scripts/train.py --config configs/train_config.yaml
    ```

    - The training process uses K-Fold Cross-Validation to ensure robust performance across dataset splits.

5. **Evaluate the model**:

    Assess the trained model’s performance:

    ```bash
    python scripts/evaluate.py --model models/checkpoints/best_model.pth
    ```

    - Evaluation metrics include Accuracy, Precision, Recall, and F1-Score, as outlined in the project goals.

## Methodology

The methodology is based on insights from Tipper et al. (2024) and tailored to the project’s objectives:

1. **Dataset**: The FaceForensics++ dataset, containing real videos and manipulated videos (Deepfakes, Face2Face, FaceSwap), is used for training and testing.
2. **Preprocessing**: Videos are processed to extract facial regions (using MTCNN) and landmarks (using dlib) from sequential frames.
3. **Model Architecture**:
    - **CNN**: Extracts spatial features (e.g., facial textures, lighting inconsistencies) from individual frames.
    - **LSTM**: Analyzes temporal dependencies across frame sequences to detect deepfake artifacts.
4. **Evaluation**: The model is evaluated on a test set with metrics such as Accuracy (targeting 85%-92%), Precision, Recall, and F1-Score.

## Results

Preliminary experiments aim to achieve detection accuracy between 85% and 92%, aligning with benchmarks from the literature (e.g., Tipper et al., 2024 report near-perfect accuracy with optimized CNN-LSTM models).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements, such as improved feature extraction, model optimizations, or additional datasets.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## References

-   Tipper, S., Atlam, H. F., & Lallie, H. S. (2024). An Investigation into the Utilisation of CNN with LSTM for Video Deepfake Detection. _Applied Sciences_, 14(21), 9754. [https://www.mdpi.com/2076-3417/14/21/9754](https://www.mdpi.com/2076-3417/14/21/9754)
-   Rossler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. _ICCV_. [https://arxiv.org/abs/1901.08971](https://arxiv.org/abs/1901.08971)
-   Kaggle FaceForensics++ Dataset: [https://www.kaggle.com/datasets/xdxd003/ff-c23](https://www.kaggle.com/datasets/xdxd003/ff-c23)
