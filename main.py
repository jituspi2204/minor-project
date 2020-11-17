import numpy as np
from flask import Flask,jsonify,request,render_template,send_from_directory,make_response
import pickle
from flask_cors import CORS, cross_origin


app = Flask('Heart Disease')
CORS(app)
model = pickle.load(open('heart_model.pkl','rb'))
@app.route('/')
def home():
    return send_from_directory('build',"index.html")

@app.route('/static/<path:path>')
def staticpath(path):
    return send_from_directory('static',path)


@app.route('/predict' ,methods=['POST'])
def ping():
    data = request.get_json(force=True)
    inputData = [np.array(list(data.values()))]
    result = int(model.predict(inputData)[0])
    print(result)
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0',port=8080)
