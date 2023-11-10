import os

from flask import request
from flask import jsonify
from flask import Flask

from ultralytics import YOLO
from apps.run_yolov8 import start_infer

app = Flask(__name__)

# load model
HOME = os.getcwd()
MODEL = f"{HOME}/server/weights/best.pt"
model = YOLO(MODEL)

@app.route('/')
def status():
    return 'connected'

# if a request POST is made on this url it runs the running function below
# CCTV영상 FPS는 30이라고 가정, 3초동안 측정된 개수의 중앙값과 timestamp return
@app.route('/predict',methods=['POST'])
def running():
    # start infer by yolo
    response = start_infer(model)
    # put the response in json and return
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)