# DeepFake Detection Model Training

This repository contains code for training a deep learning model to detect deepfake videos and images.

## Project Description

This project implements a CNN-based deepfake detection model that analyzes facial features and inconsistencies to identify manipulated media. The model is trained on a dataset of real and fake images/videos.

Key features:

-   CNN architecture optimized for deepfake detection
-   Training on both image and video data
-   Face extraction and preprocessing pipeline
-   Evaluation metrics and visualization tools

## Directory Structure

The repository is organized as follows:

```
deep-fake-detection-on-social-media/
│
├── dataset/
│   ├── raw/                # Raw dataset of real and fake media
│   ├── processed/          # Preprocessed data ready for training
│
├── models/
│   ├── checkpoints/        # Saved model checkpoints
│   ├── pretrained/         # Pretrained models for transfer learning
│
├── scripts/
│   ├── train.py            # Script for training the model
│   ├── evaluate.py         # Script for evaluating the model
│   ├── preprocess.py       # Script for data preprocessing
│
├── notebooks/
│   ├── exploration.ipynb   # Jupyter notebook for data exploration
│   ├── visualization.ipynb # Notebook for visualizing results
│
├── results/
│   ├── logs/               # Training and evaluation logs
│   ├── metrics/            # Saved evaluation metrics
│
└── README.md               # Project documentation
```

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://gitlab.com/danghaithinh/deep-fake-detection-on-social-media.git
    cd deep-fake-detection-on-social-media
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Prepare the dataset by running the preprocessing script:

    ```bash
    python scripts/preprocess.py --input data/raw --output data/processed
    ```

4. Train the model:

    ```bash
    python scripts/train.py --config configs/train_config.yaml
    ```

5. Evaluate the model:
    ```bash
    python scripts/evaluate.py --model models/checkpoints/best_model.pth
    ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
