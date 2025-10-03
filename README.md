# End-to-End ML Pipeline for Real-Time Material Classification (AI Scrap Sorting)

This project implements a complete Machine Learning pipeline for classifying scrap materials (waste) from image data, focusing on **speed, clarity, and deployment readiness** via ONNX

## 1. Project Goal & Architecture

The objective was to build a multi-class image classification system and deploy it in a simulated low-latency conveyor belt environment.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data** | TrashNet (6 Classes) | Dataset Preparation, Augmentation, and Splitting. |
| **Model** | **ResNet-18** | CNN architecture using **Transfer Learning** with ImageNet weights. |
| **Deployment** | **ONNX** Runtime | Lightweight, optimized inference script for sub-100ms classification speed. |
| **Simulation** | Python / CSV Logging | Simulates frame-by-frame classification on a conveyor with confidence thresholding and logging. |

## 2. Dataset Used: TrashNet

* **Source:**  sourced from Kaggle's "[Trashnet Dataset](https://www.kaggle.com/datasets/feyzazkefe/trashnet),"
* **Classes :** 6 classes: `Cardboard`, `Glass`, `Metal`, `Paper`, `Plastic`, `Trash`.
* **Preprocessing:** Images were resized to 224x224 and normalized using ImageNet mean/std.
* **Augmentation (Training Set):** Random flips, rotations ($\pm 15^\circ$), and color jitter were applied to enhance model robustness.
* **Data Split (Training Run):** Approximately 80% Train, 20% Validation (Test set reserved for simulation).

## 3. Model Development & Training

### Architecture

We used **ResNet-18** due to its efficiency and strong performance, making it suitable for lightweight deployment scenarios like embedded systems (e.g., Jetson Nano, Xavier).

### Training Process (Transfer Learning)

1.  **Base Model:** Loaded ResNet-18 with weights pre-trained on the massive ImageNet dataset.
2.  **Modification:** Replaced the final Fully Connected layer to map feature space to 6 output classes.
3.  **Optimization:** The pre-trained convolutional layers were initially frozen, and only the new classification head was trained. This was followed by fine-tuning the full network with a lower learning rate.

### Performance Summary (Validation Set)

| Metric | Value (from model_train.py run) |
| :--- | :--- |
| **Accuracy** | **79.21%** |
| **Precision (Macro)** | **78.05%** |
| **Recall (Macro)** | **76.72%** |

## 4. Lightweight Deployment Decisions

The final model weights (`best_model.pth`) were converted to the **ONNX (Open Neural Network eXchange)** format.

* **Decision Rationale:** ONNX provides a standardized, optimized format that significantly reduces inference latency compared to native PyTorch/TensorFlow for production/edge environments.
* **Inference Engine:** The `src/inference_engine.py` script utilizes the **ONNX Runtime** library for maximum speed and minimal memory overhead during the simulated real-time classification.
* **Observed Speed:** Average inference speed during simulation was consistently low (e.g., **~50-60 ms** per frame on a typical CPU), suitable for high-speed scrap sorting.

## 5. Folder Structure & How to Run

### Project Structure

```text
AI_SCRAP_SORTING_PIPELINE/
├── data/
│   ├── trashnet/       # <- Contains 6 class subfolders
│   └── retraining_queue/ # <- BONUS: Stores misclassified images
├── models/
│   ├── best_model.pth    # PyTorch Checkpoint
│   └── optimized_model.onnx # Lightweight Deployment model
├── results/
│   ├── simulation_log.csv # Final Simulation Output Log
│   ├── confusion_matrix.png
│   └── training_history.png
└── src/ (All source code)
```

### Setup and Execution

1.  **Setup Environment:**
    ```powershell
    python -m venv venv
    .\scrapenv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
2.  **Train the Model & Save PyTorch Checkpoint:**
    ```powershell
    python -m src.model_train
    ```
3.  **Convert to ONNX (Deployment Step):**
    ```powershell
    python -m src.inference_engine
    ```
4.  **Run the Simulated Real-Time Conveyor:**
    ```powershell
    python -m src.real_time_simulation
    ```

## 6. Optional (Bonus Points) Implementation

* **Active Learning Simulation:** The `src/real_time_simulation.py` script automatically copies any image that meets the review criteria (`confidence < 0.85` **OR** `predicted_class != true_class`) into the `data/retraining_queue` folder. This simulates preparing a batch of high-value, hard examples for the next retraining cycle.
* **Manual Override Logic:** Misclassified items are explicitly checked against the ground truth (True Class vs. Predicted Class) and flagged in the log (`manual_override_flag = TRUE`) and console (`🚨 REVIEW (MISCLASSIFIED)`).
