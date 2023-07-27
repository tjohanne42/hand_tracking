from flask import Flask,render_template,Response
import cv2
from cv_hand_tracking import CvHandTracking
# from flask_ngrok import run_with_ngrok

app = Flask(__name__)
# run_with_ngrok(app)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
hand_tracking = CvHandTracking()


def generate_frames():
    while True:
            
        ## read the camera frame
        success, img = cap.read()
        if not success:
            break

        # Hand detection / classification
        img = hand_tracking.process(img)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    # app.run(host="localhost", port=9000)
    app.run(host='localhost', port=9000,debug=False, threaded=True)