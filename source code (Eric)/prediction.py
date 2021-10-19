from flask import Flask, render_template, Response
from camera import VideoCamera
from Soundcard import TonePrediction


app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), 
                    mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def pred():
    def voice():
        Tp = TonePrediction()
        predict = Tp.get_voice()
        #print (predict)
        yield predict
    return Response(voice(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)