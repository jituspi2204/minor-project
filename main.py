import numpy as np
from flask import Flask,jsonify,request,render_template,send_from_directory,make_response
import pickle
from flask_cors import CORS, cross_origin
import keras
import keras as k
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask('Heart Disease')
CORS(app)
model = pickle.load(open('heart_model.pkl','rb'))
ckd_model = k.models.load_model('ckd.model')
@app.route('/')
def home():
    return send_from_directory('build',"index.html")

@app.route('/static/<path:path>')
def staticpath(path):
    return send_from_directory('static',path)


@app.route('/predict/heart' ,methods=['POST'])
def heart():
    data = request.get_json(force=True)
    arrData = list(data.values())
    del arrData[-1]
    inputData = [np.array(arrData)]
    result = int(model.predict(inputData)[0])
    return jsonify(result=result)

@app.route('/predict/ckd' ,methods=['POST'])
def kidney():
    data = request.get_json(force=True)
    arrData = list(data.values())
    del arrData[-1]
    print([arrData])
    result = ckd_model.predict([arrData])
    print(result[0][0])
    return jsonify(result=float(result[0][0]))

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0',port=8080)
