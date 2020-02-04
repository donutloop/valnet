from tensorflow.keras.models import model_from_json
from valnet.settings import BASE_DIR
import tensorflow_hub as hub
import numpy as np

class Predictor(object):
    def __init__(self):
        # load json and create model
        json_file = open(BASE_DIR+'/core/usecase/tensorflow_model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, custom_objects={'KerasLayer':hub.KerasLayer})
        # load weights into new model
        loaded_model.load_weights(BASE_DIR+"/core/usecase/tensorflow_model/model.h5")
        print("Loaded model from disk")
        self.model = loaded_model

    def validateAddress(self, address: str):
        values = self.model.predict(np.array([address]))
        return values[0][0]