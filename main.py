'''RNN'''
def dataProcessing_Pandas(samples):
    df = pd.read_csv('/kaggle/input/dados-para-projeto/Repartio da Produo_20240901_20240930.csv', sep=';', skiprows = 2)
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
    
    df_w = pd.read_csv('/kaggle/input/smart-meters-in-london/weather_hourly_darksky.csv')
    df_w = df_w.loc[:samples-1, ['humidity', 'apparentTemperature', 'visibility', 'icon', 'summary','precipType']]

    #label encoder summary and icon str - int
    le_icon = LabelEncoder()
    df_w['icon'] = le_icon.fit_transform(df_w['icon'])

    le_summary = LabelEncoder()
    df_w['summary'] = le_summary.fit_transform(df_w['summary'])

    le_precip = LabelEncoder()
    df_w['precipType'] = le_precip.fit_transform(df_w['precipType'])

    # One-hot encoding using pandas
    icon_oh = pd.get_dummies(df_w['icon'], prefix='icon')
    summary_oh = pd.get_dummies(df_w['summary'], prefix='summary')
    precip_oh = pd.get_dummies(df_w['precipType'], prefix='precipType')
    
    ###########  WEATHER DATA  ###########
    
    #dataframes concat
    data = pd.concat([df_w,df[['hour','Generation - Consumption']]], axis=1)
    
    #shuffling for training data
    df_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True) 

    return data, df_shuffled

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T12:50:08.843231Z","iopub.execute_input":"2024-10-21T12:50:08.843789Z","iopub.status.idle":"2024-10-21T12:50:08.858691Z","shell.execute_reply.started":"2024-10-21T12:50:08.843731Z","shell.execute_reply":"2024-10-21T12:50:08.857380Z"},"jupyter":{"source_hidden":true}}
def dataProcessing_toNumpy(df_shuffled, data, samples):
    data_train = np.array(df_shuffled)
    data_array = np.array(data)
    
    x_t = data_train[:, :data.shape[1]-1]
    y_t = data_train[:, data.shape[1]-1]
    
    x = data_array[:, :data.shape[1]-1]
    y = data_array[:, data.shape[1]-1]
    
    num_train = int((samples*0.8))
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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T12:50:08.876562Z","iopub.execute_input":"2024-10-21T12:50:08.877090Z","iopub.status.idle":"2024-10-21T12:50:08.890433Z","shell.execute_reply.started":"2024-10-21T12:50:08.877029Z","shell.execute_reply":"2024-10-21T12:50:08.889154Z"},"jupyter":{"source_hidden":true,"outputs_hidden":false}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T12:50:08.892373Z","iopub.execute_input":"2024-10-21T12:50:08.892842Z","iopub.status.idle":"2024-10-21T12:50:08.909631Z","shell.execute_reply.started":"2024-10-21T12:50:08.892791Z","shell.execute_reply":"2024-10-21T12:50:08.908153Z"},"jupyter":{"source_hidden":true}}
def prediction(model, x_test, y_test, scaler_y):
    y_pred = model.predict(x_test)
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1,1))
    
    return y_pred, y_test

# %% [code] {"execution":{"iopub.status.busy":"2024-10-22T12:18:37.551476Z","iopub.execute_input":"2024-10-22T12:18:37.551969Z","iopub.status.idle":"2024-10-22T12:18:37.590469Z","shell.execute_reply.started":"2024-10-22T12:18:37.551898Z","shell.execute_reply":"2024-10-22T12:18:37.589171Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T12:50:08.924555Z","iopub.execute_input":"2024-10-21T12:50:08.925139Z","iopub.status.idle":"2024-10-21T12:52:13.081485Z","shell.execute_reply.started":"2024-10-21T12:50:08.925082Z","shell.execute_reply":"2024-10-21T12:52:13.080112Z"},"jupyter":{"source_hidden":true}}
#Main

samples = 192 #7 days worth of data
eta = 1e-3
batch_size = 24
epochs = 1000

data, df_shuffled = dataProcessing_Pandas(samples)
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
model.save('energy_lack.h5')
