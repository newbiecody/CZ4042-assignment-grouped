import cv2
from premodel import FacialExpressionModel
import numpy as np

MOD_PATH ='face/face_mod.h5'
JSON_PATH ='face/model.json'

facec = cv2.CascadeClassifier('face/haarcascade_frontalface_default.xml')
model = FacialExpressionModel(JSON_PATH, MOD_PATH)
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        _, fr = self.video.read()
        #gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(fr, 1.3, 5)


        for (x, y, w, h) in faces:
            fc = fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (64, 64))
            roi = roi[np.newaxis, :, :,np.newaxis]
            roi = roi.reshape(1,64,64,3)
            pred = model.predict_emotion(roi)

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()