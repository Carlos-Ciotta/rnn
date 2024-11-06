'''IMPORTS'''
import pandas as pd
from GRUNetwork import GRUNetwork
import StorageSelection as ss
import DataProcessing as dp
import os

if __name__ == "__main__":
    df_es = pd.read_csv('datasets/rnn/dataset_energystorage.txt', sep=';')
    samples = 191 #7 days worth of data
    eta = 1e-3
    batch_size = 24
    epochs = 500

    data = dp.dataProcessing_Pandas(samples)
    x_train,x_test, y_train, y_test, scalerx, scalery = dp.dataProcessing_toNumpy(data, samples, 0.7, 0.3)

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

    options, necessity_time = ss.options_filter(y_pred, x_test, scalerx, df_es)
    results_df = pd.DataFrame(columns=['Hour', 
                                       'References',
                                       'Energy_Supply',
                                       'Demand', 
                                       'Total Cost',
                                       'Storages'])

    for hour in range(necessity_time.shape[0]):
        options, message, demand, indexes, energy_supply = ss.select_storage(necessity_time[hour, 0] * -1, options, hour)
        costs = ss.describe_selection(hour,indexes, energy_supply, necessity_time)

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