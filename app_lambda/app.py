import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime
import os
import logging
import boto3 

# In Lambda, logs go to CloudWatch automatically
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Model Loading ---
MODEL_S3_BUCKET = os.environ.get('MODEL_S3_BUCKET')
MODEL_S3_KEY = os.environ.get('MODEL_S3_KEY')
LOCAL_MODEL_PATH = '/tmp/pneumonia_model.onnx'

# Global variable for the ONNX session to allow reuse across invocations (for warm starts)
onnx_session = None

def download_model_from_s3(bucket, key, download_path):
    """Downloads the model from S3 to a local path in Lambda."""
    try:
        s3 = boto3.client('s3')
        logger.info(f"Downloading model from s3://{bucket}/{key} to {download_path}")
        s3.download_file(bucket, key, download_path)
        return True
    except Exception as e:
        logger.error(f"Error downloading model from S3: {e}", exc_info=True)
        return False

def load_onnx_model():
    """Loads the ONNX model, downloading from S3 if not already present locally."""
    global onnx_session
    if onnx_session is None: # Only load if not already loaded (for warm starts)
        if not os.path.exists(LOCAL_MODEL_PATH):
            if MODEL_S3_BUCKET and MODEL_S3_KEY:
                if not download_model_from_s3(MODEL_S3_BUCKET, MODEL_S3_KEY, LOCAL_MODEL_PATH):
                    raise RuntimeError("Failed to download ONNX model from S3.")
            else:
                raise EnvironmentError("Missing S3 bucket/key environment variables for model.")

        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Loading ONNX model from local path: {LOCAL_MODEL_PATH}")
            try:
                onnx_session = onnxruntime.InferenceSession(LOCAL_MODEL_PATH, providers=['CPUExecutionProvider'])
                logger.info("ONNX model loaded successfully into session.")
            except Exception as e:
                logger.error(f"Error loading ONNX model into session: {e}", exc_info=True)
                raise RuntimeError(f"Failed to load ONNX model: {e}")
        else:
            raise FileNotFoundError(f"Model file not found at {LOCAL_MODEL_PATH} after attempting download.")
    return onnx_session

# Image Preprocessing
def preprocess_image_from_base64(base64_string, target_size=(224, 224)):
    """Decodes base64 image, converts to RGB, resizes, and normalizes."""
    try:
        img_bytes = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        img_array_batch = np.expand_dims(img_array, axis=0)
        return img_array_batch
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}", exc_info=True)
        return None

# --- Lambda Handler ---
def lambda_handler(event, context):
    logger.info(f"Received event: {event}") # Log the event for debugging

    try:
        # Load model (will download from S3 on cold start if not already cached in /tmp)
        session = load_onnx_model()
        if session is None: # Should have raised an error in load_onnx_model, but double check
            raise RuntimeError("ONNX session is not available.")

        input_name = session.get_inputs()[0].name

        # API Gateway often wraps the request body in a string under 'body'
        if isinstance(event.get('body'), str):
            try:
                request_body = json.loads(event['body'])
            except json.JSONDecodeError:
                logger.error("Request body is a string but not valid JSON.")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Invalid JSON in request body'})
                }
        else:
            request_body = event.get('body', {}) # For direct Lambda test or other triggers

        if not isinstance(request_body, dict): # After potential json.loads
            logger.error(f"Request body is not a dictionary: {type(request_body)}")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Request body must be a JSON object'})
            }

        image_base64 = request_body.get('image')
        if not image_base64:
            logger.error("No 'image' field found in request body or image data is empty.")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing "image" field in JSON body (base64 encoded string)'})
            }

        # Preprocess the image
        # Assuming IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS are fixed as per training (224,224,3)
        img_array = preprocess_image_from_base64(image_base64, target_size=(224, 224))
        if img_array is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Image preprocessing failed. Check image format or content.'})
            }

        # Run inference
        logger.info(f"Running inference. Input shape: {img_array.shape}")
        result = session.run(None, {input_name: img_array.astype(np.float32)})
        raw_prediction = result[0] # Assuming model has one output
        logger.info(f"Raw ONNX Prediction output: {raw_prediction}")

        # Interpret prediction (consistent with predict_local.py)
        predicted_probability_pneumonia = float(raw_prediction[0][0])
        threshold = 0.5
        predicted_label_str = "PNEUMONIA" if predicted_probability_pneumonia > threshold else "NORMAL"

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps({
                'predicted_label': predicted_label_str,
                'probability_pneumonia': predicted_probability_pneumonia,
                'modelVersion': MODEL_S3_KEY
            })
        }

    except RuntimeError as e: # Catch specific errors from model loading
        logger.error(f"Runtime error: {e}", exc_info=True)
        return {
            'statusCode': 503, # Service Unavailable (e.g., model couldn't load)
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': f'Service initialization error: {str(e)}'})
        }
    except Exception as e:
        logger.error(f"Unhandled error in lambda_handler: {e}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': f'Internal server error: {str(e)}'})
        }