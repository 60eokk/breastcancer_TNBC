import tensorflow as tf
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    try:
        model_path = os.path.join(model_dir, 'model.h5')
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    logger.info(f"Received input: {request_body[:100]}...")  # Log first 100 chars
    try:
        return np.array(json.loads(request_body), dtype=np.float32)
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_data, model):
    try:
        logger.info(f"Input shape: {input_data.shape}")
        prediction = model.predict(input_data)
        logger.info(f"Prediction shape: {prediction.shape}")
        return prediction
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, content_type):
    try:
        return json.dumps(prediction.tolist())
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise