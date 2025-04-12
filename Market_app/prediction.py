from flask import Flask, render_template, redirect, url_for, request

import yfinance as yf
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError 
import numpy as np
import csv
import flask_firewall
print(f"Eager execution enabled: {tf.executing_eagerly()}")


# Copyright (c) 2025 Joie Harvey
# All Rights Reserved.
#
# Licensed under the All Rights Reserved. Unauthorized use or redistribution is prohibited.




app = Flask (__name__)

firewall = flask_firewall.Firewall(50,60)
firewall.add_to_whitelist('127.0.0.1', "Localhost")

firewall.startTempBlacklist_removal()
firewall.startperiodic_check()




market_model = load_model('market_prediction.h5')

symbol = 'EURUSD=X'
interval = '15m'

custom_header = [ 'Close', 'High', 'Low', 'Open', 'Volume' ]

historical_data = yf.download(symbol, period="1y", interval="1d")



historical_data.columns = custom_header


    
   

historical_data.to_csv("user_history_data.csv")

    

batch_size = 32

price_csv = pd.read_csv("user_history_data.csv")









closing_data = price_csv.Close
closing_data.loc[closing_data < 1] = 1 / closing_data[closing_data < 1]

        

close_tensor = tf.convert_to_tensor(closing_data.values, dtype=tf.float32)
tensor_size = close_tensor.shape
close_tensor = tf.reshape(close_tensor, (tensor_size[0],1))
price_ds = tf.data.Dataset.from_tensor_slices(close_tensor)
price_ds = price_ds.batch(batch_size).shuffle(100).prefetch(tf.data.experimental.AUTOTUNE) 
price_ds = price_ds.map(lambda x:(x,x))

market_model.compile(optimizer='adam', loss= 'mse')

market_model.fit(price_ds, epochs=12)

@app.route('/main', methods=['POST', 'GET'])
def main():
    is_blocked = firewall.block_access()
    if is_blocked == 403:
        return "Denied Access", 403
    if firewall.rate_limiter() == 429:
        return "Too Many Requests", 42
    return render_template('main.html')

@app.route('/')
def main_redirect():
    is_blocked = firewall.block_access()
    if is_blocked == 403:
        return "Denied Access", 403
    if firewall.rate_limiter() == 429:
        return "Too Many Requests", 429
    return redirect(url_for('main'))

@app.route('/result')
def result():
    is_blocked = firewall.block_access()
    if is_blocked == 403:
        return "Denied Access", 403
    if firewall.rate_limiter() == 429:
        return "Too Many Requests", 429
    return render_template('result.html')


@app.route('/input', methods=['POST', 'GET'])
def input():

    is_blocked = firewall.block_access()
    if is_blocked == 403:
        return "Denied Access",403
    if firewall.rate_limiter() == 429:
        return "Too Many Requests", 429
   
    global  symbol, interval

    try:
        symbol = firewall.santitize_input(request.form['symbol'])
        if firewall.identify_payloads(symbol) == 403:
            return "Malicious Activity Detected, Access permanently denied", 403
        interval = request.form['radio-option']
        if interval == '1mo':
            str_interval = 'month'
        elif interval == '1d':
            str_interval = 'day'

        
        
        current_market_data = yf.download(symbol,period='1y', interval=interval)
        last_data = current_market_data.tail(1)
        current_price = last_data['Close'].iloc[0].values[0]
        print(current_price)
        price_array = np.array([[current_price]])

        model_prediction = market_model.predict(price_array[0])

        if model_prediction > current_price:
            direction = 'Upward'
        elif model_prediction < current_price:
            direction = 'Downward'
        else:
            direction = 'Stagnantly'







        return render_template('result.html',price_prediction=model_prediction[0][0], market_direction=direction, symbol=symbol, interval=str_interval)
    except Exception:
        return redirect(url_for('main'))
    
        




if __name__ == '__main__':
    app.run(debug=True, port=5006)