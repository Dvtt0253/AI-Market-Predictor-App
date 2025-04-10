import yfinance as yf
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras import models, layers
import numpy as np


# Copyright (c) 2025 Joie Harvey
# All Rights Reserved.
#
# Licensed under the All Rights Reserved. Unauthorized use or redistribution is prohibited.

 


symbol = 'EURUSD=X' 
interval='15m' 

historical_data = yf.download(symbol, period="1y", interval="1d") 

#historical_data.to_csv("history_data.csv")


price_csv = pd.read_csv("history_data.csv") 
closing_data = price_csv.Close 
close_tensor = tf.convert_to_tensor(closing_data.values, dtype=tf.float32) 
close_tensor = tf.reshape(close_tensor, (261, 1))
close_dataset = tf.data.Dataset.from_tensor_slices(close_tensor)







batch_size = 32 

input_shape=(1,)
latent_dim = 10 

encoder = models.Sequential() 
encoder.add(layers.InputLayer(shape=input_shape)) 
encoder.add(layers.Dense(128, activation='relu')) 
encoder.add(layers.Dense(latent_dim, activation='relu'))


decoder = models.Sequential() 
decoder.add(layers.InputLayer(shape=(latent_dim,)))
decoder.add(layers.Dense(128, activation='relu'))  
decoder.add(layers.Dense(input_shape[0], activation='linear')) 

autoencoder = models.Sequential([encoder, decoder])

autoencoder.compile(optimizer='adam', loss='mean_squared_error') 


close_dataset = close_dataset.batch(batch_size).shuffle(100).prefetch(tf.data.experimental.AUTOTUNE)
close_dataset = close_dataset.map(lambda x:(x,x)) 
autoencoder.fit(close_dataset, epochs=12) 
autoencoder.save('market_prediction.h5')
while True:

    data = yf.download(symbol,period='1d', interval=interval) 

    last_data = data.tail(1)

    price = last_data['Close'].iloc[0].values[0] 
    price_input = np.array([[price]]) 

    model_prediction = autoencoder.predict(price_input[0]) 
    print(f"Current Price: {price}")
    print(f"Model's Prediction: {model_prediction}")

    time.sleep(900)








