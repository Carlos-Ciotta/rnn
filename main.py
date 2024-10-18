#Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras 

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
def dataProcessing_Pandas(samples):
    #data processing
    
    ###########  GENERATION AND CONSUMPTION DATA  ###########
    
    df = pd.read_csv('/kaggle/input/ren-data/Repartio da Produo_20240901_20240930.csv', sep=';', skiprows = 2)
    df = df.loc[:samples-1, :]
    df['Data e Hora'] = pd.to_datetime(df['Data e Hora'])
    
    df['day'] = pd.to_datetime(df['Data e Hora'], format = '%Y-%m-%d %H:%M:%S')
    df['year'] = df['Data e Hora'].dt.year
    df['month'] = df['Data e Hora'].dt.month
    df['day'] = df['Data e Hora'].dt.day
    df['hour'] = df['Data e Hora'].dt.hour
    df['minute'] = df['Data e Hora'].dt.minute
    
    
    df['Geracao'] = df['Hídrica']+df['Eólica']+df['Solar']+df['Biomassa']+df['Ondas']+df['Gás Natural - Ciclo Combinado']+df['Gás natural - Cogeração']+df['Carvão']+df['Outra Térmica']+df['Bombagem']#+df['Importação']
    df['Generation - Consumption'] = df['Geracao'] - df['Consumo']

    df_processed = df[['day','hour','minute','Generation - Consumption']]
    
    ###########  GENERATION AND CONSUMPTION DATA  ###########
    
    ###########  WEATHER DATA  ###########
    
    df_w = pd.read_csv('/kaggle/input/smart-meters-in-london/weather_daily_darksky.csv')
    df_w = df_w.loc[:samples-1, ['pressure', 'temperatureMax', 'cloudCover', 'windSpeed','dewPoint', 'icon', 'summary']]
    #df_w = df_w.loc[:samples-1, ['temperatureMax', 'cloudCover','dewPoint', 'icon', 'summary']]
    #label encoder summary and icon str - int
    le = LabelEncoder()
    df_w['icon'] = le.fit_transform(df_w['icon'])
    df_w['summary'] = le.fit_transform(df_w['summary'])
    '''
    #not using anymore
    #one_hot function to summary and icon
    icon_oh =keras.ops.one_hot(df_w['icon'], samples, axis=-1)
    summary_oh =keras.ops.one_hot(df_w['summary'], samples, axis=-1)
    
    #insert icon_one_hot and summary_one_hot
    icon_oh = pd.DataFrame(icon_oh, columns=[f'icon_{i}' for i in range(icon_oh.shape[1])])
    summary_oh = pd.DataFrame(summary_oh, columns=[f'summary_{i}' for i in range(summary_oh.shape[1])])
    
    df_w.drop(columns=['icon', 'summary'], inplace=True) #drop unused tables
    '''
    ###########  WEATHER DATA  ###########
    
    #dataframes concat
    data = pd.concat([df_w,df_processed], axis=1)
    
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
    
    num_train = int(samples*0.8)
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

    return x_train, x_test, y_train, y_test, scaler_y
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
    plt.xlabel("15 MINUTES", fontsize=12)
    plt.ylabel("Generation - Consumption", fontsize=12)
    
    # Ading legends of lines
    plt.legend()
    
    # Show Graph
    return plt.show()
#Main

samples = 800 #30 days worth of data
eta = 1e-3
batch_size = 96
epochs = 1200

data, df_shuffled = dataProcessing_Pandas(samples)
data.head()
x_train,x_test, y_train, y_test, scaler = dataProcessing_toNumpy(df_shuffled, data, samples)
#print('y_train:',y_train)
#print('x_train:',x_train)
#plt.boxplot(y_train)
#outliers_y = remove_outliers(y_train.flatten())  # flatten 2D vector -> 1D vector

# Remove outliers from samples
#x_train_clean = x_train[~outliers_y]
#y_train_clean = y_train[~outliers_y]

#plt.boxplot(y_train_clean)

model = RNN(x_train, x_test, y_train, y_test, eta, batch_size, epochs)
y_pred, y_test = prediction(model, x_test, y_test, scaler)
plot_prediction(y_pred, y_test)
