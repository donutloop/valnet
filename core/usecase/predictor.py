from tensorflow.keras.models import model_from_json
from valnet.settings import BASE_DIR
import tensorflow_hub as hub
import numpy as np
from core.models import AddressValidationHistory

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
        if not values:
            Exception("predcitor could not predict value")

        accuracy = values[0][0]
        valid = False
        if accuracy >= 0.8:
            valid = True

        addressValidationHistory = AddressValidationHistory()
        addressValidationHistory.valid = valid
        addressValidationHistory.address = address
        addressValidationHistory.accuracy = accuracy
        addressValidationHistory.save()

        return (valid, accuracy)