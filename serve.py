import tensorflow as tf
import numpy as np
import json

def model_fn(model_dir):
    model = tf.keras.models.load_model(model_dir + '/model.h5')
    return model

def input_fn(request_body, request_content_type):
    return np.array(request_body, dtype=np.float32)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())