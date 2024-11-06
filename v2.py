'''IMPORTS'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def dataProcessing_Pandas(samples):
    df = pd.read_csv('/content/Repartição da Produção_20230101_20230108.csv', sep=';', skiprows = 2)
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

    ###########  CONSUM POSTAL CODE 4200 DATA  ###########
    df_c = pd.read_csv('/content/consumos_horario_codigo_postal.csv', sep=';')
    df_c = df_c.sort_values('Data/Hora').reset_index(drop=True)
    df_c = df_c.loc[:(samples-1), :]
    df_c['Energia ativa (kWh)']=df_c['Energia ativa (kWh)']

    df['Geracao'] = df['Hídrica']+df['Eólica']+df['Solar']+df['Biomassa']+df['Ondas']+df['Gás Natural - Ciclo Combinado']+df['Gás natural - Cogeração']+df['Bombagem']#+df['Importação']
    constraint = ((df_c['Energia ativa (kWh)'].sum()) * 100) / (df['Consumo'].sum())*0.01
    df['Generation - Consumption'] = (df['Geracao']*constraint) - df_c['Energia ativa (kWh)']

    ###########  GENERATION AND CONSUMPTION DATA  ###########

    ###########  WEATHER DATA  ###########

    df_w = pd.read_csv('/content/weather_data_porto.txt',sep=';')
    df_w = df_w.loc[:samples-1, ['Temperatura',	'Weather'	,'Rain',	'Wind Velocity (km|h)']]

    #label encoder summary and icon str - int
    le_weather = LabelEncoder()
    le_weekday = LabelEncoder()
    df_w['Weather'] = le_weather.fit_transform(df_w['Weather'])
    df_c['Dia da Semana'] = le_weekday.fit_transform(df_c['Dia da Semana'])
    #dataframes concat
    data = pd.concat([df_w[['Temperatura',	'Weather'	,'Rain',	'Wind Velocity (km|h)']],df_c['Dia da Semana'],df[['day','hour','Generation - Consumption']]], axis=1)

    return data


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

def dataProcessing_toNumpy(data, samples, train_rate, test_rate):
    data_array = np.array(data)

    x = data_array[:, :data.shape[1]-1]
    y = data_array[:, data.shape[1]-1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_rate, random_state=42)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    outliers_y = remove_outliers(y_train)
    # Remove outliers from samples
    x_train = x_train[~outliers_y]
    y_train = y_train[~outliers_y]

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Defining scaler
    scalerx_train = MinMaxScaler()
    scalerx_test = MinMaxScaler()
    scalery_train = MinMaxScaler()
    scalery_test = MinMaxScaler()

    # normalizing
    x_train = scalerx_train.fit_transform(x_train)
    x_test = scalerx_test.fit_transform(x_test)

    y_train = scalery_train.fit_transform(y_train)
    y_test = scalery_test.fit_transform(y_test)

    # Reshaping to layers: (samples, time_steps, features)
    x_num = x.shape[1]
    steps = 1
    x_train = np.reshape(x_train, (x_train.shape[0],steps, x_num))
    x_test = np.reshape(x_test, (x_test.shape[0], steps,x_num))

    return x_train, x_test, y_train, y_test, scalerx_test, scalery_test
  class GRUNetwork:
    def __init__(self,x_train, x_test, y_train, y_test, eta, batch_size, epochs):
        self.model = Sequential()

        self.model.add(GRU(75, return_sequences=True, input_shape = (x_train.shape[1], x_train.shape[2]), activation ='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(GRU(50, return_sequences=True, activation ='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(GRU(35, return_sequences=False, activation ='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(20))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(1))

        self.optimizer = Adam(learning_rate=eta)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=int(epochs*0.7), restore_best_weights=True)

        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics = ['mae'])

    def train(self, x_train, y_train, epochs, batch_size, x_test, y_test):
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),callbacks=[self.early_stopping], batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test, y_test, scaler_y):
        y_pred = self.model.predict(x_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
        y_test = scaler_y.inverse_transform(y_test.reshape(-1,1))

        return y_pred, y_test

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)

    def plot_prediction(self,y_pred, y_test):
        # Figure Size
        plt.figure(figsize=(10, 6))

        # Ploting lines
        plt.plot(y_test, label="Energy Consumption", color='blue', linestyle='dashed')
        plt.plot(y_pred[:, 0], label="Prediction", color='red', alpha=0.7)

        # Title and Label
        plt.title("Real values x Predicted vales", fontsize=14)
        plt.xlabel("Hours", fontsize=12)
        plt.ylabel("Energetic Balance", fontsize=12)

        # Ading legends of lines
        plt.legend()

        # Show Graph
        return plt.show()

def options_filter(y_pred, x_test, scalerx, df_es):
    time = x_test[:, 0, :]
    time = scalerx.inverse_transform(time)
    time = time[:, x_test.shape[2]-1]
    time = np.reshape(time, (x_test.shape[0],1))

    necessity_time = np.hstack((y_pred, time))

    ref = np.array(df_es[['Ref']])
    load = np.array(df_es[['Storage_MW']])
    price = np.array(df_es[['price(E|kWh)']])
    options = np.hstack((load, price, ref))
    options = options[options[:, 1].argsort()]

    return options, necessity_time

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

          if(demand < array[index,0]):
            energy_supply.append(demand)
            indexes.append(index)
            array[index,0] = array[index,0] - demand
            demand = 0

          if(demand > array[index,0] and array[index,0]>0):
            energy_supply.append(array[index,0])
            indexes.append(index)
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

def describe_selection(indexes, energy_supply, necessity_time):
    cost = 0
    if(indexes):
      
      for index in range(len(indexes)):
        cost = cost + (necessity_time[index, 1]/100)*energy_supply[index]
      print(f"\n### HOUR : {hour} ### \nIndex:{indexes}\nEnergy Supply from each index:{energy_supply}\nCost for the stored energy in hour:{hour} is €{cost*1000}\n#########################")
    return cost*1000

if __name__ == "__main__":
    df_es = pd.read_csv('/content/dataset_energystorage.txt', sep=';')
    samples = 191 #7 days worth of data
    eta = 1e-3
    batch_size = 24
    epochs = 500

    data = dataProcessing_Pandas(samples)
    x_train,x_test, y_train, y_test, scalerx, scalery = dataProcessing_toNumpy(data, samples, 0.7, 0.3)

    rnn = GRUNetwork(x_train, x_test, y_train, y_test, eta, batch_size, epochs)

    if (os.path.exists('energy_lack.h5')):
        rnn.load_model('energy_lack.h5')
        #loss, accuracy = rnn.evaluate(x_test, y_test)
        y_pred, y_test = rnn.predict(x_test, y_test, scalery)
        rnn.plot_prediction(y_pred, y_test)
    else:
        rnn.train(x_train, y_train, epochs, batch_size, x_test,y_test)
        loss, accuracy = rnn.evaluate(x_test, y_test)
        rnn.save_model('energy_lack.h5')
        y_pred, y_test = rnn.predict(x_test, y_test, scalery)
        rnn.plot_prediction(y_pred, y_test)

    options, necessity_time = options_filter(y_pred, x_test, scalerx, df_es)
    results_df = pd.DataFrame(columns=['Hour', 
                                       'References',
                                       'Energy_Supply',
                                       'Demand', 
                                       'Total Cost',
                                       'Storages'])

    for hour in range(necessity_time.shape[0]):
        options, message, demand, indexes, energy_supply = select_storage(necessity_time[hour, 0] * -1, options, hour)
        costs = describe_selection(indexes, energy_supply, necessity_time)

        # Append the results to the DataFrame
        references = [options[i, 2] for i in indexes] 
        new_row = pd.DataFrame({'Hour': [hour], 
                                'References': [references], 
                                'Energy_Supply': [energy_supply], 
                                'Demand': [demand],
                                'Total Cost':[costs], 
                                'Storages':[options[:,0]]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    
   # print(f"\n\nUpdated values for the storages for the end of the day:\n {options[:,0]}")
