import numpy as np
import soundcard as snd

from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
import opensmile

import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QStackedWidget
from PyQt5.QtCore import QTimer

class Screen(QDialog):
    def __init__(self): 
        super(Screen, self).__init__()
        loadUi("start.ui", self)
        self.start.clicked.connect(self.gotostart)
    
    def gotostart(self):
        predict = PredictScreen()
        widget.addWidget(predict)
        widget.setCurrentIndex(widget.currentIndex()+1)

class PredictScreen(QDialog):

    mic = snd.default_microphone()
    spk = snd.default_speaker()
    data = []
    model = load_model("sm_model_0.71.hdf5")

    EMOTIONS = ['Angry', 'Bordom', 'Calm', 
                'Disgusted', 'Fearful', 'Happy', 
                'Neutral', 'Sad', 'Suprised']
    lb = LabelEncoder()
    utils.to_categorical(lb.fit_transform(EMOTIONS))

    def __init__(self):
        super(PredictScreen, self).__init__()
        self.setWindowTitle("Prediction")
        loadUi("predict.ui", self)
        self.recordbutton.clicked.connect(self.record)
        self.playbutton.clicked.connect(self.play)
        
    def record(self):
        self.info.setText("Recording...")
        QTimer.singleShot(50, self.recorded)

    def recorded(self):
        self.data = self.mic.record(samplerate = 48000, numframes=48000*5)
        self.info.setText("Recording Done")

        self.predbutton = QPushButton(self)
        self.predbutton.setText("Predict")
        self.predbutton.setGeometry(90, 560, 131,61)
        self.predbutton.setStyleSheet("border-radius:10px; font: 15pt; background-color: rgb(170, 255, 255);")
        self.predbutton.show()
        self.predbutton.clicked.connect(self.prediction)


    def play(self):
        if len(self.data) == 0:
            self.error.setText("Please record before playing.")
            QTimer.singleShot(1500, lambda: self.error.clear())
        else:
            #self.error.setStyleSheet()
            self.error.setText('<font color="black">Playing...</font>')
            QTimer.singleShot(250, lambda: self.spk.play(self.data/np.max(self.data), samplerate=48000))
            QTimer.singleShot(1500, lambda: self.error.clear())
            

    def preprocessing(self):

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        feature = smile.process_signal(self.data.T[0], 44100)

        return feature

    def prediction(self):
        features = self.preprocessing()
        features = np.expand_dims(features, axis=2)
        prediction = self.model.predict(features)
        prediction = np.argmax(prediction, axis=1)
        prediction = self.lb.inverse_transform(prediction)
        self.display_result(prediction)


    def display_result(self, prediction):
        self.predict_text.setText(prediction[0])

# Main
app = QApplication(sys.argv)    # Creating an application
screen = Screen()               # Creating an object from the app class defined
widget = QStackedWidget()       # Very useful if you are stacking different widget, switching from different screen
widget.addWidget(screen)
widget.setFixedHeight(800)
widget.setFixedWidth(1000)
widget.setWindowTitle("Prediction Application")
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")

