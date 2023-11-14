import os
from datetime import datetime

from flask import request
from flask import jsonify
from flask import Flask


from ultralytics import YOLO
from apps.run_yolov8 import start_infer

# FLASK_APP=server/app.py flask run --host=0.0.0.0 를 이용해 실행
# FLASK_APP=server/app.py nohup flask run --host=0.0.0.0 & 를 이용해 백그라운드 실행
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
    ids = request.json.get('id')
    start_time = datetime.now()
    results = []
    # start infer by yolo
    for id in ids:
        print('id:', id, '-'*50)
        yolo_res = start_infer(model, id)
        id_res = {'timestamp':start_time, 'id':id, 'tent_cnt':yolo_res}
        results.append(id_res)
    # put the response in json and return
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)