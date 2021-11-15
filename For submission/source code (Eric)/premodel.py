import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from keras.models import model_from_json


class FacialExpressionModel(object):

    EMOTION_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json, model_file):
        with open(model_json, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_file)

    def predict_emotion(self, img):
        self.pred = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTION_LIST[np.argmax(self.pred)]