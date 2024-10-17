import pandas as pd
#consumption and info dataframe
samples = 800

df_c = pd.read_csv('/kaggle/input/smart-meters-in-london/daily_dataset.csv')
df_c = df_c.loc[:samples-1, :]
df_c['day'] = pd.to_datetime(df_c['day'], format = '%Y-%m-%d')
df_c['year'] = df_c['day'].dt.year
df_c['month'] = df_c['day'].dt.month
df_c['day'] = df_c['day'].dt.day
df_c = df_c[['year', 'month', 'day', 'energy_sum']]

#weather dataframe
df_w = pd.read_csv('/kaggle/input/smart-meters-in-london/weather_daily_darksky.csv')
df_w = df_w.loc[:samples-1, ['pressure','temperatureMax', 'cloudCover', 'windSpeed','dewPoint', 'icon', 'summary']]

#creating holidays dataframe
#df_h = pd.DataFrame([0, 1, 0, 0, 1], columns = ['holiday'], dtype='int')

#label encoder str to int for one_hot function
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_w['icon'] = le.fit_transform(df_w['icon'])
df_w['summary'] = le.fit_transform(df_w['summary'])

#one_hot function to summary and icon
import tensorflow as tf

icon_oh =tf.keras.ops.one_hot(df_w['icon'], samples, axis=-1)
summary_oh =tf.keras.ops.one_hot(df_w['summary'], samples, axis=-1)

#insert icon_one_hot and summary_one_hot
icon_oh = pd.DataFrame(icon_oh, columns=[f'icon_{i}' for i in range(icon_oh.shape[1])])
summary_oh = pd.DataFrame(summary_oh, columns=[f'summary_{i}' for i in range(summary_oh.shape[1])])
df_w.drop(columns=['icon', 'summary'], inplace=True) #drop unused tables

# concat
#df = pd.concat([df_w,icon_oh, summary_oh,df_c], axis=1)
df = pd.concat([df_w, icon_oh,df_c], axis=1)

#array management

import numpy as np

data = np.array(df)
x = data[:, :df.shape[1]-1]
y = data[:, df.shape[1]-1]

num_train = int(samples*0.8)
num_test = int(samples - num_train)

# Training variables
x_train = x[0:num_train]
y_train = y[0:num_train]
y_train = np.reshape(y_train, (num_train, 1))
# Test variables
x_test = x[num_train:]
y_test = y[num_train:]
y_test = np.reshape(y_test, (num_test, 1))

from sklearn.preprocessing import MinMaxScaler
# Defining scaler
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_num = x.shape[1]
steps = 1
# normalizing
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Reshaping to LSTM: (samples, time_steps, features)
x_train = np.reshape(x_train, (num_train,steps, x_num))
x_test = np.reshape(x_test, (num_test, steps,x_num))

import matplotlib.pyplot as plt
# remove outliers using (IQR)
def remove_outliers(data, threshold=1.5):
    #Q1 e Q3
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    inferior_limit = Q1 - threshold * IQR
    upper_limit = Q3 + threshold * IQR
    
    # Identifica as amostras que são outliers
    outliers = (data < inferior_limit) | (data > upper_limit)
    
    return outliers

outliers_y = remove_outliers(y_train.flatten())  # flatten 2D vector -> 1D vector

# Remove outliers from samples
x_train_clean = x_train[~outliers_y]
y_train_clean = y_train[~outliers_y]

#plt.boxplot(y_train_clean)

#RNN Development
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam
import keras

model = Sequential()

#Input Layer
model.add(LSTM(75, input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = True))
model.add(Dropout (0.3))

#Input Layer
model.add(LSTM(50, return_sequences = False))
model.add(Dropout (0.3))

#Hidden Layer
model.add(Dense(20))
model.add(Dropout (0.3))

#output layer
model.add(Dense(1))

#optmizer
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['mae'])
#model.summary()

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2000, restore_best_weights=True)

model.fit(x_train_clean, y_train_clean, epochs=4000, validation_data=(x_test, y_test),callbacks=[early_stopping], batch_size=32)

y_pred = model.predict(x_test)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test.reshape(-1,1))


#plot
# Figure Size
plt.figure(figsize=(10, 6))

# Ploting lines
plt.plot(y_test, label="Energy Consumption", color='blue', linestyle='dashed')
plt.plot(y_pred[:, 0], label="Prediction", color='red', alpha=0.7)

# Title and Label
plt.title("Real values x Predicted vales", fontsize=14)
plt.xlabel("Days", fontsize=12)
plt.ylabel("Energy Consumption (kWh)", fontsize=12)

# Ading legends of lines
plt.legend()

# Show Graph
plt.show()
