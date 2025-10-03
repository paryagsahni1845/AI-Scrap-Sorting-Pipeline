# src/real_time_simulation.py (Stage 4 & Bonus)
import os
import time
import csv
from datetime import datetime
from glob import glob
import random
import shutil

# Import necessary components
from src.inference_engine import InferenceEngine, preprocess_image, ONNX_MODEL_PATH
from src.data_prep import DATA_DIR, CLASSES

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.85
SIMULATION_DELAY_SEC = 0.1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "../results/simulation_log.csv")
RETRAIN_QUEUE_DIR = os.path.join(BASE_DIR, "../data/retraining_queue")

def setup_log_file(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_id', 'timestamp', 'true_class', 'predicted_class', 
            'confidence', 'flagged_low_confidence', 'manual_override_flag', 'inference_time_ms'
        ])

def get_simulation_frames():
    """Get sample images for simulation (first 50 shuffled frames)."""
    image_paths = []
    for cls in CLASSES:
        paths = glob(os.path.join(DATA_DIR, cls, '*.jpg'))
        image_paths.extend([(p, cls) for p in paths])
    random.shuffle(image_paths)
    return image_paths[:50]

def run_simulation():
    print("--- Starting Real-Time Scrap Classification Simulation ---")
    
    setup_log_file(RESULTS_CSV)

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ERROR: ONNX model not found at {ONNX_MODEL_PATH}. Please convert it first!")
        return
    
    engine = InferenceEngine(onnx_path=ONNX_MODEL_PATH)
    image_data = get_simulation_frames()
    print(f"Simulating {len(image_data)} frames at {1/SIMULATION_DELAY_SEC} FPS...")

    for i, (path, true_class) in enumerate(image_data):
        frame_id = os.path.basename(path)
        time.sleep(SIMULATION_DELAY_SEC)

        # --- Inference ---
        start_time = time.time()
        try:
            image_tensor = preprocess_image(path)
            predicted_class, confidence = engine.classify_frame(image_tensor)
        except Exception as e:
            print(f"Error classifying frame {frame_id}: {e}")
            continue
        inference_time_ms = (time.time() - start_time) * 1000

        # --- Flags ---
        flag_low_confidence = confidence < CONFIDENCE_THRESHOLD
        manual_flag = predicted_class != true_class

        # --- Bonus: Active Learning / Retraining Queue ---
        if flag_low_confidence or manual_flag:
            os.makedirs(RETRAIN_QUEUE_DIR, exist_ok=True)
            queue_class_dir = os.path.join(RETRAIN_QUEUE_DIR, true_class)
            os.makedirs(queue_class_dir, exist_ok=True)
            shutil.copy(path, os.path.join(queue_class_dir, frame_id))
            if manual_flag:
                print(f"  [BONUS] 🚨 MISCLASSIFIED: True={true_class.upper()} added to retraining queue.")
            elif flag_low_confidence:
                print(f"  [BONUS] ⚠️ LOW CONFIDENCE added to retraining queue.")

        # --- Console Output ---
        status = "✅ ACCEPTED"
        if flag_low_confidence:
            status = "⚠️ LOW CONFIDENCE"
        if manual_flag:
            status = "🚨 REVIEW (MISCLASSIFIED)"

        print(f"\n[{i+1}/{len(image_data)}] Frame: {frame_id} ({true_class.upper()})")
        print(f"  -> Prediction: {predicted_class.upper()}")
        print(f"  -> Confidence: {confidence:.4f} | Status: {status}")
        print(f"  -> Time: {inference_time_ms:.2f} ms")

        # --- CSV Logging ---
        with open(RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                true_class,
                predicted_class,
                f"{confidence:.4f}",
                flag_low_confidence,
                manual_flag,
                f"{inference_time_ms:.2f}"
            ])

    print(f"\n--- Simulation Complete. Results saved to {RESULTS_CSV} ---")

if __name__ == '__main__':
    run_simulation()
