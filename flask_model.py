from flask import Flask,render_template
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask import Flask, request, url_for
#from flask.ext.uploads import UploadSet, configure_uploads, SCRIPTS

flask_model = Flask(__name__)
model = pickle.load(open('pickle_file.pkl', 'rb'))

@flask_model.route('/')
def home():
	return render_template('home.html')
	
@flask_model.route('/predict',methods=["POST","GET"])
def predict():
  a=[]        
  if request.method == "POST":
    dtd = request.form['dtd']
    age = request.form['age']
    station = request.form['station']
    store = request.form['store']
    lat = request.form['lat']
    log = request.form['log']
    a.append([dtd,age,station,store,lat,log])
    final_features = np.asarray(a)
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
  return render_template('home.html', prediction_text='Unit Area of House Price is {}'.format(output))

if __name__ == "__main__":
    flask_model.run(debug=True)
