#Ei jaga datasetti training, testing, validation
#Vaid jatan niimoodi, sest teen treeningu ja testingu walk-forward validtionis

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import ta
import time

start_time = time.time()

files = ["CPI.csv", "FEDFUNDS.csv", "SavingsRate.csv", "UNRATE.csv"]

df_stock = yf.Ticker("VTR").history(start="1995-01-01", end="2019-01-01", interval="1d").reset_index()[
    ["Date", "Open", "Close", "High", "Low", "Volume"]]

dates_for_files = []
for i in df_stock["Date"]:
    i = str(i)
    i = i[:7]
    dates_for_files.append(i)

data_dict = {}
dates = []
value_list = []
#loen failide andmed üldandmestikku

for file in files:
    final_value_list = []
    df = pd.read_csv(file)
    temporary_dict = df.to_dict()
    for values in temporary_dict.values():
        for i in values.values():
            if type(i) == str:
                dates.append(i[:7])
            else:
                value_list.append(i)
    for i in range(0, len(dates)):
        data_dict[dates[i]] = value_list[i]

    for i in dates_for_files:
        final_value_list.append(data_dict[i])
    df_stock[file] = final_value_list

df_stock = df_stock.drop("Date", axis=1)

#lisan tehnilised indikaatorid andmestikku
"""
df_stock["rsi"] = ta.momentum.rsi(df_stock["Close"], window=14)
df_stock["ATR"] = ta.volatility.average_true_range(df_stock["High"], df_stock["Low"], df_stock["Close"])
df_stock["bb_upper"] = ta.volatility.bollinger_hband(df_stock["Close"])
df_stock["bb_middle"] = ta.volatility.bollinger_mavg(df_stock["Close"])
df_stock["bb_lower"] = ta.volatility.bollinger_lband(df_stock["Close"])
df_stock["sma_10"] = ta.trend.sma_indicator(df_stock["Close"], window=10)
df_stock["sma_20"] = ta.trend.sma_indicator(df_stock["Close"], window=20)
df_stock["macd"] = ta.trend.macd(df_stock["Close"])
"""
df_stock = df_stock.dropna()
print(df_stock)

x_data = df_stock.values
y_data = df_stock["Close"].values

print(len(x_data))


y_data = np.reshape(y_data, (-1, 1))
#scalen andmed 0, ja 1 vahel
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_x_data = scaler.fit_transform(x_data)
scaled_y_data = scaler.fit_transform(y_data)
print(scaled_x_data)
print(scaled_y_data)

x_train = []
y_train = []
for i in range(len(x_data)-2):
    x_train.append(scaled_x_data[i:i+2])
    y_train.append(scaled_y_data[i+2])
print(x_train)
print(y_train)

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = np.ravel(y_train)#muudan y_train 1d -ks

#hyperparameter search
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
print(x_train.shape)
print(y_train.shape)
model = SVR()

parameters = {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], "kernel": ["rbf"], "gamma": [0.01, 0.1, 1.0, 10], "epsilon": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(model, parameters, cv=5, verbose=1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

predictions = []
actual_values = []
print(len(x_train))

#walk forward validation
for i in range(len(x_train)-40):
    #preparing training data
    x_train_i = x_train[i:i+40]
    y_train_i = y_train[i:i+40]

    x_test = x_train[i+40]
    x_test = x_test.reshape((1, -1))

    print(x_test.shape)


    best_model.fit(x_train_i, y_train_i)
    print(i)
    pred = best_model.predict(x_test)
    pred = pred.reshape((-1, 1))
    pred = scaler.inverse_transform(pred)
    predictions.append(pred[0][0]) #0 ja 0 sest prediction on array ja niimoodi saame katte ennustuse
    actual_values.append(df_stock["Close"].iloc[i+40])

predictions = np.array(predictions)
actual_values = np.array(actual_values)
print(actual_values)
print(predictions)


#mudeli mõõdikud
mse = np.mean((predictions - actual_values) ** 2)
print("MSE: "+ str(mse))
rmse = np.sqrt(mse)
print("RMSE: "+ str(rmse))
mae = np.mean(np.abs(predictions - actual_values))
print("MAE: "+ str(mae))
mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
print("MAPE: "+ str(mape))

end_time = time.time()
all_time = end_time - start_time
print("Programmi jooksmise aeg: " + str(all_time))

plt.plot(actual_values, label="Actual")
plt.plot(predictions, label="Predictions")
plt.xlabel("Time step")
plt.ylabel("Closing price")
plt.legend()
plt.show()








