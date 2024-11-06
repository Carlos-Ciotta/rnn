import numpy as np
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

def describe_selection(hour,indexes, energy_supply, necessity_time):
    cost = 0
    if(indexes):
      
      for index in range(len(indexes)):
        cost = cost + (necessity_time[index, 1]/100)*energy_supply[index]
      print(f"\n### HOUR : {hour} ### \nIndex:{indexes}\nEnergy Supply from each index:{energy_supply}\nCost for the stored energy in hour:{hour} is €{cost*1000}\n#########################")
    return cost*1000

__All__ = [[options_filter, select_storage, describe_selection]]