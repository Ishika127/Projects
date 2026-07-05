from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load saved model files
with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model/kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Read form values
    txn_type = request.form['type']
    acct_type = request.form['acct_type']
    time_of_day = request.form['time_of_day']

    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])
    unusuallogin = float(request.form['unusuallogin'])
    step = float(request.form['step'])

    # Encode categorical features (order must match training: type, Acct type, Time of day)
    cat_df = pd.DataFrame([[txn_type, acct_type, time_of_day]],
                           columns=['type', 'Acct type', 'Time of day'])
    encoded_cats = encoder.transform(cat_df)

    # Scale numeric features (order must match training)
    num_df = pd.DataFrame([[amount, oldbalanceOrg, newbalanceOrig,
                             oldbalanceDest, newbalanceDest, unusuallogin, step]],
                          columns=['amount', 'oldbalanceOrg', 'newbalanceOrig',
                                   'oldbalanceDest', 'newbalanceDest', 'unusuallogin', 'step'])
    scaled_nums = scaler.transform(num_df)

    # Combine and predict cluster
    X = np.hstack([scaled_nums, encoded_cats])
    cluster = kmeans.predict(X)[0]

    return render_template('result.html', cluster=cluster)


if __name__ == '__main__':
    app.run(debug=True)
