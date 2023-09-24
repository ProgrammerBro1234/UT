import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

start_time = time.time()

files = ["CPI.csv", "FEDFUNDS.csv", "SavingsRate.csv", "UNRATE.csv"]

df_stock = yf.Ticker("NCN1T.TL").history(start="2020-03-03", end="2023-06-01", interval="1d").reset_index()[["Date", "Open", "Close", "High", "Low", "Volume"]]

dates_for_files = []
for i in df_stock["Date"]:
    i = str(i)
    i = i[:7]
    dates_for_files.append(i)

data_dict = {}
dates = []
value_list = []
"""
#loeb failid sisse
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
df_stock[file] = final_value_list"""

df_stock = df_stock.drop("Date", axis=1)

x_data = df_stock["Close"].values
y_data = df_stock["Close"].values

x_data, y_data = x_data.reshape((-1, 1)), y_data.reshape((-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_x_data = scaler.fit_transform(x_data)
scaled_y_data = scaler.fit_transform(y_data)

x_train, y_train = [], []

for i in range(len(x_data)-2):
    x_train.append(scaled_x_data[i:i+2])
    y_train.append(scaled_y_data[i+2])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))

#positional encoding
encoding_array = x_train
for pos in range(x_train.shape[0]):
    for i in range(0, x_train.shape[1], 2):
        funk = pos / np.power(10000, i / x_train.shape[1])
        encoding_array[pos, i] = np.sin(funk)
        encoding_array[pos, i+1] = np.cos(funk)

encoded_x_train = encoding_array + x_train

encoded_x_train = encoded_x_train.reshape((encoded_x_train.shape[0], encoded_x_train.shape[1], 1))
print(encoded_x_train.shape)

#building model
input = tf.keras.layers.Input(shape=(encoded_x_train.shape[1], encoded_x_train.shape[2])) #shape tuleb 3D sest treenimise ajal batch_size annab 1D juurde

lstm1 = tf.keras.layers.LSTM(units=16, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input)
lstm2 = tf.keras.layers.LSTM(units=128, return_sequences=True)(lstm1)

add_and_norm = tf.keras.layers.LayerNormalization()(lstm2)

multi_head = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=128)(add_and_norm, add_and_norm, add_and_norm)

add_and_norm = tf.keras.layers.LayerNormalization()(multi_head + input)

feed_forward = tf.keras.layers.Dense(units=128, activation="relu")(add_and_norm)

add_and_norm = tf.keras.layers.LayerNormalization()(add_and_norm + feed_forward)

output = tf.keras.layers.Dense(units=1)(add_and_norm)

model = tf.keras.Model(inputs= input, outputs=output)

model.compile(optimizer="adam", loss="mse")

model.fit(x_train, y_train, epochs=60, batch_size=32)

#walk forward validation

predictions = []
actual_values = []
print(len(x_train))
for i in range(len(x_train)-40):
    #preparing training data
    x_train_i = x_train[i:i+40]
    y_train_i = y_train[i:i+40]

    #print(x_train_i.shape)
    #print(y_train_i.shape)
    #preparing testing data
    x_test = x_train[i+40]
    x_test = x_test.reshape((1, -1))

    print(x_test.shape)


    model.fit(x_train_i, y_train_i, epochs=60)
    print(i)
    pred = model.predict(x_test)
    pred = pred.reshape((-1, 1))
    pred = scaler.inverse_transform(pred)
    predictions.append(pred[0][0]) #0 ja 0 sest prediction on array ja niimoodi saame katte ennustuse
    actual_values.append(df_stock["Close"].iloc[i+40])

predictions_for_calc = np.array(predictions)
actual_values_for_calc = np.array(actual_values)
print(actual_values)
print(predictions)

#erinevad mõõdikud

mse = np.mean((predictions_for_calc - actual_values_for_calc) ** 2)
print("MSE: "+ str(mse))
rmse = np.sqrt(mse)
print("RMSE: "+ str(rmse))
mae = np.mean(np.abs(predictions_for_calc - actual_values_for_calc))
print("MAE: "+ str(mae))
mape = np.mean(np.abs((actual_values_for_calc - predictions_for_calc) / actual_values_for_calc)) * 100
print("MAPE: "+ str(mape))

end_time = time.time()
all_time = end_time - start_time
print(all_time)

predictions = np.array(predictions)
actual_values = np.array(actual_values)

plt.plot(actual_values, label="Actual")
plt.plot(predictions, label="Predictions")
plt.xlabel("Time step")
plt.ylabel("Closing price")
plt.legend()
plt.show()



