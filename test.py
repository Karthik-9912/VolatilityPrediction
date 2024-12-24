import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM #class for LSTM training
from math import sqrt
import matplotlib.pyplot as plt
from keras.models import model_from_json
import pickle
import os
from keras.layers import Dropout

sc = MinMaxScaler(feature_range = (0, 1))

dataset = pd.read_csv('Dataset/CLX.csv')
dataset["Date"]= pd.to_datetime(dataset["Date"])
dataset["Year"]= dataset['Date'].dt.year

temp = dataset.values
Y = temp[:,5:6]

dataset = dataset.drop(["Adj Close"], axis=1)
dataset = dataset.drop(["Volume"], axis=1)
dataset = dataset.values
X = dataset[:,1:dataset.shape[1]]

X = sc.fit_transform(X)
Y = sc.fit_transform(Y)

#training with LSTM algorithm
X_train = []
y_train = []
for i in range(10, 250):
    X_train.append(X[i-10:i, 0:X.shape[1]])
    y_train.append(Y[i, 0])    
X_train, y_train = np.array(X_train), np.array(y_train)
if os.path.exists('model/lstm_model.json'):
    with open('model/lstm_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        regressor = model_from_json(loaded_model_json)
    json_file.close()
    regressor.load_weights("model/lstm_model_weights.h5")
    regressor._make_predict_function()   
else:
    #training with LSTM algorithm and saving trained model and LSTM refrence assigned to regression variable
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 1000, batch_size = 8)
    regressor.save_weights('model/lstm_model_weights.h5')            
    model_json = regressor.to_json()
    with open("model/lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
#performing prediction on test data    
predict_growth = regressor.predict(X_train)
predict_growth = sc.inverse_transform(predict_growth)
predict_growth = predict_growth.ravel()
y_train = y_train.reshape(y_train.shape[0],1)
labels = sc.inverse_transform(y_train)
labels = labels.ravel()
print("LSTM Predicted Growth: "+str(predict_growth))
print("\nOriginal Grwoth: "+str(labels))
#calculating LSTM MSE
lstm_rmse = mean_squared_error(labels,predict_growth)

print("\nLSTM Root Mean Square Error: "+str(lstm_rmse))
#plotting LSTM predicted and original values
plt.plot(labels, color = 'red', label = 'Original Stock Prices')
plt.plot(predict_growth, color = 'green', label = 'LSTM Predicted Price')
plt.title('LSTM Stock Prices Forecasting')
plt.xlabel('Test Data')
plt.ylabel('Forecasting Prices')
plt.legend()
plt.show()
