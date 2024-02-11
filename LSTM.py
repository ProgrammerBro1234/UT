import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters as hp
start_time = time.time()

files = ["CPI.csv", "FEDFUNDS.csv", "SavingsRate.csv", "UNRATE.csv"]

df_stock = yf.Ticker("KR").history(start="1995-01-01", end="2024-02-26", interval="1d").reset_index()[
    ["Date", "Open", "Close", "High", "Low", "Volume"]]



dates_for_files = []
for i in df_stock["Date"]:
    i = str(i)
    i = i[:7]
    dates_for_files.append(i)

data_dict = {}
dates = []
value_list = []

for file in files:
    final_value_list = []
    df = pd.read_csv(file, error_bad_lines=False)
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
#print(df_stock)

print(df_stock)
df_stock = df_stock.drop("Date", axis=1)

x_data = df_stock.values
y_data = df_stock.values

number_mis_on_eespool = 9

print(x_data.shape)
print(y_data.shape)
#x_data = np.reshape(x_data, (-1, 1))
#y_data = np.reshape(y_data, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_x_data = scaler.fit_transform(x_data)
scaled_y_data = scaler.fit_transform(y_data)
print(x_data.shape)
print(y_data.shape)

print(x_data.shape)


extra_x_test = scaled_x_data[-2:]
extra_x_test = extra_x_test.reshape(1, 2, number_mis_on_eespool)
x_train = []
y_train = []
for i in range(len(x_data)-2):
    x_train.append(scaled_x_data[i:i+2])
    y_train.append(scaled_y_data[i+2])

#print(x_train)
#print(y_train)

x_train = np.array(x_train)
print(x_train.shape)
y_train = np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], 2, x_train.shape[2])
y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
print(x_train.shape)
print(y_train.shape)

#hyperparameter search
def model_builder(hp):
    model = tf.keras.Sequential()
    num_layers = hp.Int("num_layers", min_value=1, max_value=100, step=1)
    model.add(tf.keras.layers.LSTM(hp.Int("tegelikult_esimene", min_value=5, max_value=150, step=5), input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    for i in range(num_layers):

        model.add(tf.keras.layers.LSTM(hp.Int("units_" + str(i), min_value=5, max_value=150, step=5), return_sequences=True))


    model.add(tf.keras.layers.Dense(hp.Int("units_" + str(num_layers), min_value=5, max_value=100, step=5)))
    model.add(tf.keras.layers.Dense(units=number_mis_on_eespool))
    model.compile(tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=[0.1, 0.001, 0.0001, 0.00001])), loss="mse")
    return model

tuner = RandomSearch(
    model_builder,
    objective="val_loss",
    max_trials=999,
    executions_per_trial=1,
    directory='C:/Users/Kasutaja/PycharmProjects/Hyperparameetrid/',
    project_name='C:/Users/Kasutaja/PycharmProjects/HyperparameetridLogs/')


tuner.search(x_train, y_train, epochs=70, validation_split=0.1, batch_size=128)

#get best hyperparameters

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
#build and compile the best model

best_model = tuner.hypermodel.build(best_hp)


print(best_model.summary())

predictions = []
actual_values = []

#walk forward validation
for i in range(len(x_train)-40):
    #preparing training data
    x_train_i = x_train[i:i+40]
    y_train_i = y_train[i:i+40]


    #y_train_i = y_train_i.reshape((40, 1, number_mis_on_eespool))
    #preparing testing data
    x_test = x_train[i+40].reshape(1, 2, number_mis_on_eespool)
    print(x_train_i.shape)
    print(y_train_i.shape)

    print(x_test.shape)


    best_model.fit(x_train_i, y_train_i, epochs=100, verbose = 0)
    print(i)
    pred = best_model.predict(x_test)
    print(pred.shape)
    #pred = pred.reshape((-1, 1))
    pred = pred.reshape((2, number_mis_on_eespool))
    print(pred.shape)
    pred = scaler.inverse_transform(pred)
    predictions.append(pred[0][0])#0 ja 0 sest prediction on array ja niimoodi saame katte ennustuse
    print(pred[0][0])
    actual_values.append(df_stock["Close"][i+40])
    #print(x_test)




#mudeli mõõdikud
predictions_metrics = np.array(predictions)
actual_values_metrics = np.array(actual_values)

mse = np.mean((predictions_metrics - actual_values_metrics) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions_metrics - actual_values_metrics))
mape = np.mean(np.abs((actual_values_metrics - predictions_metrics) / actual_values_metrics)) * 100



print(extra_x_test.shape)
esimene_pred = best_model.predict(extra_x_test)
print(esimene_pred.shape)
esimene_extra_prediction = esimene_pred.reshape((2, number_mis_on_eespool))
esimene_extra_ennustus = scaler.inverse_transform(esimene_extra_prediction)
predictions.append(esimene_extra_ennustus[0][0])

#teine
"""
esimene_pred = esimene_pred.reshape((1, 2, number_mis_on_eespool))
teine_extra_pred = best_model.predict(esimene_pred)
teine_pred = teine_extra_pred.reshape((2, number_mis_on_eespool))
print(teine_pred.shape)
teine_extra_ennustus = scaler.inverse_transform(teine_pred)
predictions.append(teine_extra_ennustus[0][0])

arr = esimene_extra_prediction.reshape((1, -1))
teine_pred = teine_extra_pred.reshape((1, 2, number_mis_on_eespool))
kolmas_extra_pred = best_model.predict(teine_pred)
kolmas_pred = kolmas_extra_pred.reshape((2, number_mis_on_eespool))
print(kolmas_pred.shape)
kolmas_extra_ennustus = scaler.inverse_transform(kolmas_pred)
predictions.append(kolmas_extra_ennustus[0][0])
"""
muutus = (predictions[-1] - predictions[-2]) / predictions[-1] *100
print("Muutus: ", muutus)

predictions = np.array(predictions)
actual_values = np.array(actual_values)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)

end_time = time.time()
all_time = end_time - start_time
print("Programmi jooksmise aeg: " + str(all_time))
#predictions = np.array(predictions)
#actual_values = np.array(actual_values)
# Plot the predictions and actual values
plt.plot(actual_values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.xlabel("Time Step")
plt.ylabel("Close Price")
plt.legend()
plt.show()





"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler
import time
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters as hp
start_time = time.time()

files = ["CPI.csv", "FEDFUNDS.csv", "SavingsRate.csv", "UNRATE.csv"]

df_stock = yf.Ticker("KR").history(start="1995-01-01", end="2023-12-08", interval="1d").reset_index()[
    ["Date", "Open", "Close", "High", "Low", "Volume"]]



dates_for_files = []
for i in df_stock["Date"]:
    i = str(i)
    i = i[:7]
    dates_for_files.append(i)

data_dict = {}
dates = []
value_list = []

for file in files:
    final_value_list = []
    df = pd.read_csv(file, error_bad_lines=False)
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
#print(df_stock)

#print(df_stock)
df_stock = df_stock.drop("Date", axis=1)

x_data = df_stock["Close"].values
y_data = df_stock["Close"].values

number_mis_on_eespool = 1

print(x_data.shape)
print(y_data.shape)
x_data = np.reshape(x_data, (-1, 1))
y_data = np.reshape(y_data, (-1, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_x_data = scaler.fit_transform(x_data)
scaled_y_data = scaler.fit_transform(y_data)
print(x_data.shape)
print(y_data.shape)

print(x_data.shape)


extra_x_test = scaled_x_data[-2:]
extra_x_test = extra_x_test.reshape(1, 2, number_mis_on_eespool)
x_train = []
y_train = []
for i in range(len(x_data)-2):
    x_train.append(scaled_x_data[i:i+2])
    y_train.append(scaled_y_data[i+2])




#print(x_train)
#print(y_train)

x_train = np.array(x_train)
print(x_train.shape)
y_train = np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], 2, x_train.shape[2])
print(x_train.shape)
print(y_train.shape)

#hyperparameter search
def model_builder(hp):
    model = tf.keras.Sequential()
    num_layers = hp.Int("num_layers", min_value=1, max_value=100, step=1)
    model.add(tf.keras.layers.LSTM(hp.Int("tegelikult_esimene", min_value=5, max_value=150, step=5), input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
    for i in range(num_layers):

        model.add(tf.keras.layers.LSTM(hp.Int("units_" + str(i), min_value=5, max_value=150, step=5), return_sequences=True))


    model.add(tf.keras.layers.Dense(hp.Int("units_" + str(num_layers), min_value=5, max_value=100, step=5)))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=[0.1, 0.001, 0.0001, 0.00001])), loss="mse")
    return model

tuner = RandomSearch(
    model_builder,
    objective="val_loss",
    max_trials=999,
    executions_per_trial=1,
    directory='C:/Users/Kasutaja/PycharmProjects/Hyperparameetrid/',
    project_name='C:/Users/Kasutaja/PycharmProjects/HyperparameetridLogs/')


tuner.search(x_train, y_train, epochs=70, validation_split=0.1, batch_size=128)

#get best hyperparameters

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
#build and compile the best model

best_model = tuner.hypermodel.build(best_hp)


print(best_model.summary())

predictions = []
actual_values = []

#walk forward validation
for i in range(len(x_train)-40):
    #preparing training data
    x_train_i = x_train[i:i+40]
    y_train_i = y_train[i:i+40]


    #y_train_i = y_train_i.reshape((40, 1, number_mis_on_eespool))
    #preparing testing data
    x_test = x_train[i+40].reshape(1, 2, number_mis_on_eespool)
    print(x_train_i.shape)
    print(y_train_i.shape)

    print(x_test.shape)


    best_model.fit(x_train_i, y_train_i, epochs=100, verbose = 0)
    print(i)
    pred = best_model.predict(x_test)
    print(pred.shape)
    pred = pred.reshape((-1, 1))
    print(pred.shape)
    pred = scaler.inverse_transform(pred)
    predictions.append(pred[0][0]) #0 ja 0 sest prediction on array ja niimoodi saame katte ennustuse
    actual_values.append(df_stock["Close"][i+40])
    #print(x_test)




#mudeli mõõdikud
predictions_metrics = np.array(predictions)
actual_values_metrics = np.array(actual_values)

mse = np.mean((predictions_metrics - actual_values_metrics) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions_metrics - actual_values_metrics))
mape = np.mean(np.abs((actual_values_metrics - predictions_metrics) / actual_values_metrics)) * 100



print(extra_x_test.shape)
esimene_pred = best_model.predict(extra_x_test)
print(esimene_pred.shape)
esimene_extra_prediction = esimene_pred.reshape((-1, 1))
esimene_extra_ennustus = scaler.inverse_transform(esimene_extra_prediction)
predictions.append(esimene_extra_ennustus[0][0])

#teine

esimene_pred = esimene_pred.reshape((1, 2, number_mis_on_eespool))
teine_extra_pred = best_model.predict(esimene_pred)
teine_pred = teine_extra_pred.reshape((1, -1))
print(teine_pred.shape)
teine_extra_ennustus = scaler.inverse_transform(teine_pred)
predictions.append(teine_extra_ennustus[0][0])

arr = esimene_extra_prediction.reshape((1, -1))
teine_pred = teine_extra_pred.reshape((1, 2, number_mis_on_eespool))
kolmas_extra_pred = best_model.predict(teine_pred)
kolmas_pred = kolmas_extra_pred.reshape((1, -1))
print(kolmas_pred.shape)
kolmas_extra_ennustus = scaler.inverse_transform(kolmas_pred)
predictions.append(kolmas_extra_ennustus[0][0])

muutus = (predictions[-1] - predictions[-2]) / predictions[-1] *100
print("Muutus: ", muutus)

predictions = np.array(predictions)
actual_values = np.array(actual_values)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Absolute Percentage Error (MAPE):", mape)

end_time = time.time()
all_time = end_time - start_time
print("Programmi jooksmise aeg: " + str(all_time))
#predictions = np.array(predictions)
#actual_values = np.array(actual_values)
# Plot the predictions and actual values
plt.plot(actual_values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.xlabel("Time Step")
plt.ylabel("Close Price")
plt.legend()
plt.show()
"""



