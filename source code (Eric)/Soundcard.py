import soundcard as snd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils, models
from pickle import load
import features
from statistics import mode

EMOTIONS = ['Angry', 'Disgusted', 
                'Fearful', 'Happy', 
                'Neutral', 'Sad', 
                'Suprised']
lb = LabelEncoder()
utils.to_categorical(lb.fit_transform(EMOTIONS))
scaler = load(open('tone/scaler.pkl', 'rb'))
model = models.load_model('tone/2605.h5',custom_objects={'f1_m' : features.f1_m})
RATE = 44100            # Samples per second in Hz, default 44100

class TonePrediction(object):
    def __init__(self):
        #self.spk = snd.default_speaker()
        self.mic = snd.default_microphone()
    def get_voice(self):
        data = self.mic.record(numframes=55125, samplerate=RATE)

        pred1 = features.get_features(data.T[0],RATE)
        pred2 = features.get_features(data.T[1],RATE)

        pred1 = scaler.transform(pred1)
        pred1 = np.expand_dims(pred1, axis=2)

        pred2 = scaler.transform(pred2)
        pred2 = np.expand_dims(pred2, axis=2)

        result1 = model.predict(pred1)
        result1 = np.argmax(result1, axis=1)
        result1 = lb.inverse_transform(result1)
        print(result1)

        result2 = model.predict(pred2)
        result2 = np.argmax(result2, axis=1)
        result2 = lb.inverse_transform(result2)
        print(result2)
        
        result1 = result1.tolist()
        res = [x for x in set(result1) if result1.count(x) > 1]
        
        return res[0]