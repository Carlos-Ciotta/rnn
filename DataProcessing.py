from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def dataProcessing_Pandas():
    df = pd.read_csv('datasets/rnn/Repartição da Produção_20230101_20230131.csv', sep=';', skiprows = 2)
    df = df.loc[:, :] #the data comes with 15 minutes break, so i have to pick 4 times more data
    df['Data e Hora'] = pd.to_datetime(df['Data e Hora'])

    df['day'] = pd.to_datetime(df['Data e Hora'], format = '%Y-%m-%d %H:%M:%S')
    df['year'] = df['Data e Hora'].dt.year
    df['month'] = df['Data e Hora'].dt.month
    df['day'] = df['Data e Hora'].dt.day
    df['hour'] = df['Data e Hora'].dt.hour
    df['minute'] = df['Data e Hora'].dt.minute

    df= df[(df['Data e Hora'].dt.hour != 00)]#cleaning the 15 minute break
    df = df[(df['Data e Hora'].dt.minute == 0)] 
    df = df.reset_index(drop=True) 

    ###########  CONSUM POSTAL CODE 4200 DATA  ###########
    df_c = pd.read_csv('datasets/rnn/consumos_horario_codigo_postal.csv', sep=';')
    df_c = df_c.sort_values('Data/Hora').reset_index(drop=True)
    df_c = df_c.loc[:, :]

    df_c['Data/Hora'] = pd.to_datetime(df_c['Data/Hora'])

    df_c['hour'] = pd.to_datetime(df_c['Data/Hora'], format = '%Y-%m-%d %H:%M:%S')
    df_c['hour'] = df_c['Data/Hora'].dt.hour
    df_c['minute'] = df_c['Data/Hora'].dt.minute

    df_c = df_c[(df_c['Data/Hora'].dt.hour != 0)]#cleaning the 30 minute break
    df_c = df_c.reset_index(drop=True) #reset index so the concat stays with 198 lines

    df_c['Energia ativa (kWh)']=df_c['Energia ativa (kWh)']*0.001 # transforming from kWh to MWh
    df['Geracao'] = df['Hídrica']+df['Eólica']+df['Solar']+df['Ondas']+df['Bombagem']
    constraint = ((df_c['Energia ativa (kWh)'].sum()) * 100) / (df['Consumo'].sum())*0.01 # percentage of energy that goes to the location from df_c
    df['Generation - Consumption'] = (df['Geracao']*constraint) - df_c['Energia ativa (kWh)']

    ###########  WEATHER DATA AND WEEKDAY DATA  ###########

    df_w = pd.read_csv('datasets/rnn/weather_data_porto_month.txt',sep=';')
    df_w = df_w.loc[:, ['Time','Temp','Weather','Humidity']]#data comes with 30 minutes interval, so I have to pick up twice data
    df_w['Time'] = df_w['Time'].apply(lambda x: x.zfill(5)) #turns times like 1:00 into 01:00
    df_w['Time'] = pd.to_datetime(df_w['Time'])

    df_w['hour'] = pd.to_datetime(df_w['Time'], format = '%H:%M')
    df_w['hour'] = df_w['Time'].dt.hour
    df_w['minute'] = df_w['Time'].dt.minute

    df_w = df_w[(df_w['Time'].dt.minute == 0)]#cleaning the 30 minute break
    df_w = df_w.reset_index(drop=True) #reset index so the concat stays with 198 lines

    #label encoder summary and icon str - int
    le_weather = LabelEncoder()
    le_weekday = LabelEncoder()
    df_w['Weather'] = le_weather.fit_transform(df_w['Weather'])
    df_c['Dia da Semana'] = le_weekday.fit_transform(df_c['Dia da Semana'])

    '''weather_one_hot = pd.get_dummies(df_w['Weather'])
    df_w = df_w.drop(['Weather', 'hour','minute', 'Time'], axis = 1)
    df_w = df_w.join(weather_one_hot)'''
    df_w = df_w.drop(['hour','minute', 'Time'], axis = 1)
    df_boolean = df_w.select_dtypes(include=['bool'])
    # Convertendo as colunas booleanas para (1, 0)
    df_w[df_boolean.columns] = df_boolean.astype(int)

    weekday_one_hot = pd.get_dummies(df_c['Dia da Semana'])
    df_c = df_c.drop('Dia da Semana', axis = 1)
    df_c = df_c.join(weekday_one_hot)

    df_boolean = df_c.select_dtypes(include=['bool'])
    # Convertendo as colunas booleanas para (1, 0)
    df_c[df_boolean.columns] = df_boolean.astype(int)

    #dataframes concat
    data = pd.concat([df_w,df_c.iloc[:,-7:],df[['Generation - Consumption']]], axis=1)
    data = data.loc[:(df_w.shape[0]-1), :]
    
    return data, df_w.shape[0]


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

__All__ = [dataProcessing_Pandas,remove_outliers,dataProcessing_toNumpy]
