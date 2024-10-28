'''IMPORTS'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree

from itertools import combinations
'''RNN TRAINING'''
def dataProcessing_Pandas(samples):
    df = pd.read_csv('/content/Repartio da Produo_20240901_20240930.csv', sep=';', skiprows = 2)
    df = df.loc[:(samples-1)*4, :] #the data comes with 15 minutes break, so i have to pick 4 times more data
    df['Data e Hora'] = pd.to_datetime(df['Data e Hora'])

    df['day'] = pd.to_datetime(df['Data e Hora'], format = '%Y-%m-%d %H:%M:%S')
    df['year'] = df['Data e Hora'].dt.year
    df['month'] = df['Data e Hora'].dt.month
    df['day'] = df['Data e Hora'].dt.day
    df['hour'] = df['Data e Hora'].dt.hour
    df['minute'] = df['Data e Hora'].dt.minute

    df= df[df['Data e Hora'].dt.minute == 0]#cleaning the 15 minute break
    df = df.reset_index(drop=True) #reset index so the concat stays with 198 lines

    df['Geracao'] = df['Hídrica']+df['Eólica']+df['Solar']+df['Biomassa']+df['Ondas']+df['Gás Natural - Ciclo Combinado']+df['Gás natural - Cogeração']+df['Carvão']+df['Outra Térmica']+df['Bombagem']#+df['Importação']
    df['Generation - Consumption'] = df['Geracao'] - df['Consumo']

    ###########  GENERATION AND CONSUMPTION DATA  ###########

    ###########  WEATHER DATA  ###########

    df_w = pd.read_csv('/content/weather_hourly_darksky.csv')
    df_w = df_w.loc[:samples-1, ['humidity', 'apparentTemperature', 'visibility', 'icon', 'summary','precipType']]

    #label encoder summary and icon str - int
    le_icon = LabelEncoder()
    df_w['icon'] = le_icon.fit_transform(df_w['icon'])

    le_summary = LabelEncoder()
    df_w['summary'] = le_summary.fit_transform(df_w['summary'])

    le_precip = LabelEncoder()
    df_w['precipType'] = le_precip.fit_transform(df_w['precipType'])

    # One-hot encoding using pandas
    '''icon_oh = pd.get_dummies(df_w['icon'], prefix='icon')
    summary_oh = pd.get_dummies(df_w['summary'], prefix='summary')'''

    ###########  WEATHER DATA  ###########

    #dataframes concat
    data = pd.concat([df_w[['humidity', 'apparentTemperature', 'visibility','precipType','summary','icon']],df[['day','hour','Generation - Consumption']]], axis=1)

    #shuffling for training data
    df_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

    return data, df_shuffled

def dataProcessing_toNumpy(df_shuffled, data, samples):
    data_train = np.array(df_shuffled)
    data_array = np.array(data)

    x_t = data_train[:, :data.shape[1]-1]
    y_t = data_train[:, data.shape[1]-1]

    x = data_array[:, :data.shape[1]-1]
    y = data_array[:, data.shape[1]-1]

    num_train = int((samples*0.875))
    num_test = int(samples - num_train)

    # Training variables
    x_train = x_t[0:num_train]
    y_train = y_t[0:num_train]
    y_train = np.reshape(y_train, (num_train, 1))
    # Test variables
    x_test = x[num_train:]
    y_test = y[num_train:]
    y_test = np.reshape(y_test, (num_test, 1))

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

    # Reshaping to layers: (samples, time_steps, features)
    x_train = np.reshape(x_train, (num_train,steps, x_num))
    x_test = np.reshape(x_test, (num_test, steps,x_num))

    return x_train, x_test, y_train, y_test, scaler_y, scaler_x

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

def RNN(x_train, x_test, y_train, y_test, eta, batch_size, epochs):
    model = Sequential()

    #Input Layer
    model.add(GRU(100, input_shape = (x_train.shape[1], x_train.shape[2]), return_sequences = True))
    model.add(Dropout (0.5))

    #Hidden Layer
    model.add(GRU(75, return_sequences = True))
    model.add(Dropout (0.5))

    #Hidden Layer
    model.add(GRU(50, return_sequences = False))
    model.add(Dropout (0.5))

    #Hidden Layer
    model.add(Dense(20))
    model.add(Dropout (0.5))

    #output layer
    model.add(Dense(1))

    #optmizer
    optimizer = Adam(learning_rate=eta)

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics = ['mae'])
    #model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)

    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),callbacks=[early_stopping], batch_size=batch_size)

    return model

def prediction(model, x_test, y_test, scaler_y):
    y_pred = model.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1,1))

    return y_pred, y_test

def plot_prediction(y_pred, y_test):
    # Figure Size
    plt.figure(figsize=(10, 6))

    # Ploting lines
    plt.plot(y_test, label="Energy Consumption", color='blue', linestyle='dashed')
    plt.plot(y_pred[:, 0], label="Prediction", color='red', alpha=0.7)

    # Title and Label
    plt.title("Real values x Predicted vales", fontsize=14)
    plt.xlabel("Horas", fontsize=12)
    plt.ylabel("Generation - Consumption", fontsize=12)

    # Ading legends of lines
    plt.legend()

    # Show Graph
    return plt.show()

#Main

samples = 192 #7 days worth of data
eta = 1e-3
batch_size = 24
epochs = 1500

data, df_shuffled = dataProcessing_Pandas(samples)
x_train,x_test, y_train, y_test, scaler, scalerx = dataProcessing_toNumpy(df_shuffled, data, samples)


#outliers_y = remove_outliers(y_train.flatten())  # flatten 2D vector -> 1D vector
# Remove outliers from samples
#x_train_clean = x_train[~outliers_y]
#y_train_clean = y_train[~outliers_y]

#plt.boxplot(y_train_clean)

model = RNN(x_train, x_test, y_train, y_test, eta, batch_size, epochs)
y_pred, y_test = prediction(model, x_test, y_test, scaler)
plot_prediction(y_pred, y_test)
model.save('energy_lack.h5')

'''ENERGY STORAGE SELECTION'''
def array_treatment(y_pred, x_test, scalerx, df_es):
    energy_pred = y_pred
    time = x_test[:, 0, :]
    time = scalerx.inverse_transform(time)
    time = time[:, 7]
    time = np.reshape(time, (39,1))

    necessity_time = np.hstack((energy_pred, time))
    necessity_oneDay = necessity_time[16:, :]

    ref = np.array(df_es[['Ref']])
    e_available = np.array(df_es[['Storage_MW']])
    price = np.array(df_es[['price(E|kWh)']])
    eAvailable_cost = np.hstack((e_available, price, ref))
    array = eAvailable_cost[eAvailable_cost[:, 1].argsort()]

    return array, necessity_oneDay

def select_storage(demand, array, hour):
    first_value = array[0, 0]
    index = 0
    indexes = []
    energy_supply = []

    if (demand< 0):
      message = str(f"No need to select a Storage, demand less than zero!. Demand:{demand}, Hour:{hour}")
      return array, message, demand, indexes, energy_supply
    else:
      if first_value < demand:
        aux = demand
        while((demand > 0) and (index <= array.shape[0])):
          if(array[index, 0]>0):
            indexes.append(index)

          if(demand < array[index,0]):
            if(array[index,0]!=0):
              energy_supply.append(demand)
            
            array[index,0] = array[index,0] - demand
            demand = 0
          else:
            if(array[index,0]!=0):
              energy_supply.append(array[index,0])

            demand = demand - array[index,0]
            array[index,0] = 0
          index = index+1
          message = str(f"Storages Selected (index): {indexes}, Demand:{aux}, Hour:{hour}")

      else:
        energy_supply.append(array[0,0])
        aux = demand
        array[0,0] = array[0,0] - demand
        demand = 0
        message = str(f"Storages Selected (index): {indexes}, Demand:{aux}, Hour:{hour}")

    return array, message, demand, indexes, energy_supply

'''RNN USAGE'''
df_es = pd.read_csv('/content/dataset_energystorage.txt', sep=';')
samples = 192
model = load_model('energy_lack.h5')
data, df_shuffled = dataProcessing_Pandas(samples)
x_train,x_test, y_train, y_test, scaler, scalerx = dataProcessing_toNumpy(df_shuffled, data, samples)
y_pred, y_test = prediction(model, x_test, y_test, scaler)

array, necessity_oneDay = array_treatment(y_pred, x_test, scalerx, df_es)

for hour in range(necessity_oneDay.shape[0]):
    array, message, demand, indexes, energy_supply = select_storage(necessity_oneDay[hour,0]*-1, array, hour)
    #print(message)
    if(indexes):
      cost = 0
      for index in range(len(indexes)):
        cost = cost + (necessity_oneDay[index, 1]/100)*energy_supply[index]
      print(f"\n### HOUR : {hour} ### \nIndex:{indexes}\nEnergy Supply from each index:{energy_supply}\nCost for the stored energy in hour:{hour} is €{cost*1000}\n#########################")

print(f"\n\nUpdated values for the storages for the end of the day:\n {array[:,0]}")
