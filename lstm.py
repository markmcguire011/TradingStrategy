import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import math
from sklearn.metrics import mean_squared_error

bitcoin = np.loadtxt('bitcoin_price.txt')
scaler = MinMaxScaler()
df = scaler.fit_transform(np.array(bitcoin).reshape(-1,1))

train_size = int(len(df)*0.65)
test_size = len(df) - train_size
train_data,test_data = df[0:train_size,:],df[train_size:len(df),:1]

def create_dataset(dataset, time_step = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
                   a = dataset[i:(i+time_step),0]
                   dataX.append(a)
                   dataY.append(dataset[i + time_step,0])
    return np.array(dataX),np.array(dataY)

# calling the create dataset function to split the data into 
# input output datasets with time step 100
time_step = 100
X_train,Y_train =  create_dataset(train_data,time_step)
X_test,Y_test =  create_dataset(test_data,time_step)

MODEL_PATH = "/Users/markmcguire/Downloads/MATH371-MCM-Example/saved_models/lstm_stock_model.keras"  # Path to save/load the model
EPOCHS = 100
BATCH_SIZE = 64

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_or_train_model(x_train, y_train, input_shape):
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        print("Loading model from file...")
        model = load_model(MODEL_PATH)
    else:
        print("Training new model...")
        model = build_model(input_shape)
        # Model checkpoint to save only the best model during training
        checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, Y_test), callbacks=[checkpoint], verbose = 1)
    return model

model = get_or_train_model(X_train, Y_train, (X_train.shape[1], 1))

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# transform to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

print(math.sqrt(mean_squared_error(Y_train,train_predict)))
print(math.sqrt(mean_squared_error(Y_test,test_predict)))

look_back = 100

trainPredictPlot = np.empty_like(df)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back : len(train_predict)+look_back,:] = train_predict

testPredictPlot = np.empty_like(df)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict)+(look_back)*2 + 1 : len(df) - 1,:] = test_predict

plt.plot(scaler.inverse_transform(df), label = "Actual")
plt.plot(trainPredictPlot, label = "Training")
plt.plot(testPredictPlot, label = "Prediction")
plt.legend()
plt.show()