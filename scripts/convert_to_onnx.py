import tensorflow as tf
import tf2onnx
import yaml
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "config" / "params.yaml"
MODELS_DIR = PROJECT_ROOT / "models"


def load_params():
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
    return params

def convert_keras_to_onnx(keras_model_name, onnx_model_name):
    params = load_params()
    keras_model_path = MODELS_DIR / keras_model_name
    onnx_model_path = MODELS_DIR / onnx_model_name

    if not keras_model_path.exists():
        logging.error(f"Keras model not found at: {keras_model_path}")
        return

    logging.info(f"Loading Keras model from: {keras_model_path}")
    try:
        model = tf.keras.models.load_model(str(keras_model_path), compile=False) # Often good to set compile=False
        model.summary()
    except Exception as e:
        logging.error(f"Error loading Keras model: {e}")
        return

    # Define the input signature for the ONNX model
    # This should match the input shape your model expects.
    input_signature = [
        tf.TensorSpec(
            (None, params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'], params['NUM_CHANNELS']),
            tf.float32,
            name="input_image" # Name the input tensor
        )
    ]

    logging.info(f"Converting Keras model to ONNX. Input signature: {input_signature}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,
            output_path=str(onnx_model_path)
        )
        logging.info(f"ONNX model conversion successful and saved to: {onnx_model_path}")

    except Exception as e:
        logging.error(f"Error during ONNX conversion: {e}", exc_info=True) # Added exc_info for more details
        return



if __name__ == "__main__":

    BEST_KERAS_MODEL_FILENAME = "pneumonia_cnn_model_ad052d6d734e4ac9b9b0f9cfcb9d3a76.keras"
    ONNX_OUTPUT_FILENAME = "pneumonia_model.onnx"

    convert_keras_to_onnx(BEST_KERAS_MODEL_FILENAME, ONNX_OUTPUT_FILENAME)


