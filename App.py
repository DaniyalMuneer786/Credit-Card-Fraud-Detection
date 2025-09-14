from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

loaded_model = joblib.load('Random_Forest_Model.pkl')

@app.route('/')
def form():
    return render_template('Form.html')

# @app.route('/')
# def style():
    # return render_template('Form.css')


@app.route('/predict', methods=['POST' , 'GET'])
def predict():

    cc_num = float(request.form['cc_num'])
    merchant = float(request.form['merchant'])
    category = float(request.form['category'])
    amt = float(request.form['amt'])
    
    # Validation on city
    city = str(request.form['city'])
    if len(city) > 3:
        raise ValueError("Length of city should not be greater than 3")
    
    # Validation on state
    state = str(request.form['state'])
    if len(state) > 2:
        raise ValueError("Length of state should not be greater than 2")
    
    lat = float(request.form['lat'])
    long = float(request.form['long'])
    trans_date = float(request.form['trans_date'])
    trans_num = float(request.form['trans_num'])

    input_data = pd.DataFrame([[cc_num, merchant, category, amt, city, state, lat, long, 
                               trans_date, trans_num]], columns=['Credit Card Number', 'Merchant',
                                'Category', 'Amount', 'City', 'State', 'Latitude', 'Longitude',
                                'Transaction Date', 'Transaction Number'])
    
    prediction = loaded_model.predict(input_data)
    
    
    class_mapping = {0: 'Not Fraud', 1: 'Fraud'}

    predicted_class = class_mapping[prediction[0]]


    return render_template('Result.html', cc_num=cc_num, merchant=merchant, category=category, amt=amt, city=city, 
                    state=state, lat=lat, long=long, trans_date=trans_date, trans_num=trans_num, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)

