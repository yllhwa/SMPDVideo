# SMP Challenge 2025 - Video Track

This repository contains the official implementation of our solution for the SMP Challenge 2025 (Video Track).

We provide reproducible code for training and inference, along with necessary intermediate results, model weights, and other required files.

Required files can be downloaded from the following link:
[https://drive.google.com/file/d/1yRAyvXIsqpwyHpMLTfq15BV7VqUHE5zC/view?usp=sharing](https://drive.google.com/file/d/1yRAyvXIsqpwyHpMLTfq15BV7VqUHE5zC/view?usp=sharing)

## Data Preparation

1. Dataset preparation is required before running the code. The dataset should be structured as follows:

```
└─dataset                    # Dataset directory
    ├─test                   # Raw test videos
    └─train                  # Raw train videos
```

2. All model weights should be placed in the `save_model` folder.
3. Preprocessed data should be placed in the `data` folder.

## Inference
Simply run `x_inference.py`; upon completion, it will generate the `submission_final.csv` file.

## Train
### Data Preprocessing
Run the following scripts in order to complete data preprocessing:

- get_data_feat.py
- extract_video_embedding.py
- extract_video_embedding_pca.py

### Training
Just run the following scripts to train the models:
- `x_train_lightgbm.py`: Trains a boosting model with lightgbm.
- `x_train_tabnet.py`: Trains a TabNet model.
- `x_train.py`: Trains a boosting model(mainly used).

