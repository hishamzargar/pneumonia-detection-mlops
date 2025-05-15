import onnxruntime
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
import logging

from data_preprocessing import load_and_preprocess_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "config" / "params.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
# Path to a sample image for testing
SAMPLE_IMAGE_RELATIVE_PATH = "data/raw/chest_xray/test/NORMAL/NORMAL2-IM-0012-0001.jpeg"


def load_params():
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
    return params

def predict_with_onnx(onnx_model_path_str, image_path_str, target_dims):
    logging.info(f"Loading ONNX model from: {onnx_model_path_str}")
    try:
        session = onnxruntime.InferenceSession(onnx_model_path_str, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        logging.info(f"ONNX Model Input name: {input_name}, Output names: {[out.name for out in session.get_outputs()]}")
    except Exception as e:
        logging.error(f"Error loading ONNX model or getting input/output names: {e}")
        return None

    logging.info(f"Loading and preprocessing image: {image_path_str}")
    # Use the same preprocessing function used during training
    img_array = load_and_preprocess_image(image_path_str, target_size=target_dims)

    if img_array is None:
        logging.error("Image preprocessing failed.")
        return None

    # Add batch dimension if not already present (model expects NCHW or NHWC)
    # load_and_preprocess_image returns (H, W, C), so we need (1, H, W, C)
    if img_array.ndim == 3:
        img_array_batch = np.expand_dims(img_array, axis=0)
    elif img_array.ndim == 4 and img_array.shape[0] == 1:
        img_array_batch = img_array
    else:
        logging.error(f"Unexpected image array dimension: {img_array.ndim}")
        return None

    logging.info(f"Running inference with ONNX model. Input shape: {img_array_batch.shape}")
    try:
        result = session.run(None, {input_name: img_array_batch.astype(np.float32)})
    except Exception as e:
        logging.error(f"Error during ONNX inference: {e}")
        return None

    return result

if __name__ == "__main__":
    params = load_params()
    onnx_model_filename = "pneumonia_model.onnx"
    onnx_model_full_path = str(MODELS_DIR / onnx_model_filename)

    sample_image_full_path = str(PROJECT_ROOT / SAMPLE_IMAGE_RELATIVE_PATH)

    # Check if ONNX model exists
    if not Path(onnx_model_full_path).exists():
        logging.error(f"ONNX model not found at {onnx_model_full_path}. Run convert_to_onnx.py first.")
        exit()

    # Check if sample image exists
    if not Path(sample_image_full_path).exists():
        logging.error(f"Sample image not found at {sample_image_full_path}. Update SAMPLE_IMAGE_RELATIVE_PATH.")
        exit()


    target_image_dims = (params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'])
    prediction_output = predict_with_onnx(onnx_model_full_path, sample_image_full_path, target_image_dims)

    if prediction_output is not None:
        # The output is usually a list of numpy arrays.
        # For a single output model, it's prediction_output[0]
        raw_prediction = prediction_output[0]
        logging.info(f"Raw ONNX Prediction output: {raw_prediction}") # e.g., [[0.02]] for sigmoid

        # Interpret the prediction
        # For binary classification with sigmoid (outputting a single value per sample):
        # Value close to 0 -> NORMAL (class 0)
        # Value close to 1 -> PNEUMONIA (class 1)
        predicted_probability_pneumonia = raw_prediction[0][0] # Assuming batch size 1, and 1 output unit
        threshold = 0.5
        predicted_label = "PNEUMONIA" if predicted_probability_pneumonia > threshold else "NORMAL"
        
        logging.info(f"Predicted probability for PNEUMONIA: {predicted_probability_pneumonia:.4f}")
        logging.info(f"Predicted Label: {predicted_label}")