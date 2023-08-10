import os
from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'flask_app/static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

model = tf.keras.applications.ResNet50(weights='imagenet')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def perform_detection(img):
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = tf.expand_dims(x, axis=0)
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=2)[0]
    return decoded_predictions

def gen_frames():  # Generate frame by frame from camera
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            predictions = perform_detection(img_resized)

            for i, (_, label, prob) in enumerate(predictions):
                cv2.putText(frame, f"{label}: {prob:.2f}", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
