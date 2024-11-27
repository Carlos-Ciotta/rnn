import pandas as pd
from GRUNetwork import GRUNetwork
import StorageSelection as ss
import DataProcessing as dp
import os

eta = 1e-3
batch_size = 32
epochs = 3000
if __name__ == "__main__":
    data, samples = dp.dataProcessing_Pandas()
    x_train,x_test, y_train, y_test, scalerx, scalery = dp.dataProcessing_toNumpy(data, samples, 0.8, 0.2)

    rnn = GRUNetwork(x_train, x_test, y_train, y_test, eta, batch_size, epochs)

    if(os.path.exists('energy_lack4.h5')):
        rnn.load_model('energy_lack4.h5')
        loss, accuracy = rnn.evaluate(x_test, y_test)
        print(f'loss : {loss}\naccuracy : {accuracy}')
        y_pred, y_test = rnn.predict(x_test, y_test, scalery)
        rnn.plot_prediction(y_pred, y_test)
    else:
        rnn.train(x_train, y_train, epochs, batch_size, x_test,y_test)
        loss, accuracy = rnn.evaluate(x_test, y_test)
        print(f'loss : {loss}\naccuracy : {accuracy}')
        rnn.save_model('energy_lack4.h5')
        y_pred, y_test = rnn.predict(x_test, y_test, scalery)
        rnn.plot_prediction(y_pred, y_test)

    df_es = pd.read_csv('datasets/rnn/dataset_energystorage.txt', sep=';')
    options, necessity_time = ss.options_filter(y_pred, x_test, scalerx, df_es)
    aux_o = options
    results_df = pd.DataFrame(columns=['Hour', 
                                       'References',
                                       'Energy_Supply',
                                       'Demand_end', 
                                       'Cost Storage',
                                       'Cost Import'])
    cost_i = []
    cost_s = []
    energy_s = []
    for hour in range(0, 24):
        options, message, demand, indexes, energy_supply = ss.select_storage(necessity_time[hour, 0] * -1, options, hour)
        cost_storage, cost_import = ss.calculate_cost(indexes, options[:, 1], energy_supply, demand)
        cost_i.append(cost_import)
        cost_s.append(cost_storage)
        if(energy_supply is None):
            energy_s.append(0)
        else:
            energy_s.append(sum(energy_supply))
        # Append the results to the DataFrame
        references = [options[indexes[i], 2] for i in range(0,len(indexes))] 
        new_row ={              'Hour': hour, 
                                'References': references, 
                                'Energy_Supply': energy_supply, 
                                'Demand_end': demand,
                                'Cost Storage':cost_storage,
                                'Cost Import':cost_import}
        results_df.loc[len(results_df)] = new_row

    ss.plot_accuracy(y_test, energy_s)
    ss.plot_prices(cost_i, cost_s, (24))
    print(results_df)
    #print(f"\n\nUpdated values for the storages for the end of the day:\n {options[:,0]}")
