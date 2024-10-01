
def model_fn(model_dir):
    model = tf.keras.models.load_model(model_dir + '/1')
    return model

def input_fn(request_body, request_content_type):
    import numpy as np
    return np.array(request_body, dtype=np.float32)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    import json
    return json.dumps(prediction.tolist())

