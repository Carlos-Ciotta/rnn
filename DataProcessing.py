from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def dataProcessing_Pandas(samples):
    df = pd.read_csv('datasets/rnn/Repartição da Produção_20230101_20230108.csv', sep=';', skiprows = 2)
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
    df_c = pd.read_csv('datasets/rnn/consumos_horario_codigo_postal.csv', sep=';')
    df_c = df_c.sort_values('Data/Hora').reset_index(drop=True)
    df_c = df_c.loc[:(samples-1), :]
    df_c['Energia ativa (kWh)']=df_c['Energia ativa (kWh)']*0.001

    df['Geracao'] = df['Hídrica']+df['Eólica']+df['Solar']+df['Biomassa']+df['Ondas']+df['Gás Natural - Ciclo Combinado']+df['Gás natural - Cogeração']+df['Bombagem']#+df['Importação']
    constraint = ((df_c['Energia ativa (kWh)'].sum()) * 100) / (df['Consumo'].sum())*0.01
    df['Generation - Consumption'] = (df['Geracao']*constraint) - df_c['Energia ativa (kWh)']

    ###########  GENERATION AND CONSUMPTION DATA  ###########

    ###########  WEATHER DATA  ###########

    df_w = pd.read_csv('datasets/rnn/weather_data_porto.txt',sep=';')
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

__All__ = [dataProcessing_Pandas,remove_outliers,dataProcessing_toNumpy]