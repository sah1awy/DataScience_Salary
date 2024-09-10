import flask 
from flask import Flask,jsonify,request
import json
import pickle
from data_input import data_in
import numpy as np



def load_model():
    f = 'D:\\Data_Science_Salary\\DataScience_Salary\\FlaskAPI\\models\\rf.pkl'
    with open(f,'rb') as pickled:
        model = pickle.load(pickled)
    return model

app = Flask(__name__)
@app.route('/predict',methods=['Get'])
def predict():
    request_json = request.get_json()
    x = request_json['input']
    # print(x)
    x_in = np.array(x).reshape(1,-1)
    model = load_model()
    prediction = model.predict(x_in)[0]
    response = json.dumps({"response":prediction})
    return response,200

if __name__ == "__main__":
    app.run(debug=True)
