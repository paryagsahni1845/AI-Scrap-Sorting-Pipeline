# =================================================================
# src/inference_engine.py (Stage 3: Lightweight Deployment)
# =================================================================
import torch
import torch.onnx
import onnxruntime
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from src.data_prep import IMAGE_SIZE, NORM_MEAN, NORM_STD, CLASSES
from src.model_train import setup_model
import torch.nn.functional as F  # for softmax

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_model.pth")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "../models/optimized_model.onnx")

# --- Inference transforms (must match test_transforms) ---
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

def convert_to_onnx(model_path=MODEL_PATH, onnx_path=ONNX_MODEL_PATH):
    """Convert trained PyTorch model to ONNX for deployment."""
    # 1. Setup model
    model = setup_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # 2. Dummy input
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    try:
        # 3. Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"✅ Model successfully converted to ONNX and saved at {onnx_path}")
    except Exception as e:
        print(f"❌ Error during ONNX conversion: {e}")

class InferenceEngine:
    """Lightweight ONNX inference engine for single-frame classification."""
    def __init__(self, onnx_path=ONNX_MODEL_PATH):
        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def classify_frame(self, image_tensor):
        """
        Run inference on one image.
        :param image_tensor: torch tensor (1, 3, H, W)
        :return: (predicted_class, confidence)
        """
        input_data = image_tensor.detach().cpu().numpy()
        logits = self.session.run(None, {self.input_name: input_data})[0]
        probabilities = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        predicted_index = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0, predicted_index]
        predicted_class = CLASSES[predicted_index]
        return predicted_class, float(confidence)

def preprocess_image(image_path):
    """Load and preprocess image from disk for inference."""
    img = Image.open(image_path).convert("RGB")
    return INFERENCE_TRANSFORMS(img).unsqueeze(0)  # (1, 3, H, W)

if __name__ == "__main__":
    # Convert model to ONNX
    convert_to_onnx()
