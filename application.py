# Import modules and packages
from flask import Flask, request, render_template
import pickle
import numpy as np
from scipy.spatial import distance

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/test', methods=['GET', "POST"])
def test():
    # Main page
    return render_template('test.html')


@application.route('/upload', methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        img = request.files['file'].read()

        # Make prediction
        preds = model_predict(img)
        return preds
    return 'OK'





@application.route('/', methods=['POST'])
def get_input_values():
    val = request.form['my_form']


@application.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return 'The URL /predict is accessed directly. Go to the main page firstly'

    if request.method == 'POST':
        input_val = request.form

        if input_val != None:
            # collecting values
            vals = []
            for key, value in input_val.items():
                vals.append(float(value))

        # Calculate Euclidean distances to freezed centroids
        with open('freezed_centroids.pkl', 'rb') as file:
            freezed_centroids = pickle.load(file)

        assigned_clusters = []
        l = []  # list of distances

        for i, this_segment in enumerate(freezed_centroids):
            dist = distance.euclidean(*vals, this_segment)
            l.append(dist)
            index_min = np.argmin(l)
            assigned_clusters.append(index_min)

        return render_template(
            'predict.html', result_value=f'Segment = #{index_min}'
            )



if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80, debug=True)
