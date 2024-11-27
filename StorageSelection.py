import numpy as np
import matplotlib.pyplot as plt

def options_filter(y_pred, x_test, scalerx, df_es):
    time = x_test[:, 0, :] #get all the hour times from the dataset
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

  if (demand<= 0):
    message = str(f"No need to select a Storage, demand less than zero!. Demand:{demand}, Hour:{hour}")
    return array, message, demand, indexes, energy_supply
  else:
    if (first_value < demand):
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
      
      message = str(f"Demand:{aux}, Hour:{hour}")
      return array, message, demand, indexes, energy_supply
    else:
      energy_supply.append(demand)
      array[0,0] = array[0,0] - demand
      demand = 0
      indexes.append(index)
      message = str(f"Demand: 0, Hour:{hour}")
      
      return array, message, demand, indexes, energy_supply

def calculate_cost(indexes, options, energy_supply, demand):
    cost_s = 0
    cost_i=0
    if(demand>0):
      cost_i = cost_i + demand * 45
    for index in range(len(indexes)):
      cost_s = cost_s + (options[indexes[index]]) * energy_supply[index]
    return cost_s, cost_i #print(f"\n### HOUR : {hour} ### \nIndex:{indexes}\nEnergy Supply from each index:{energy_supply}\nCost for the stored energy in hour:{hour} is €{cost}\n#########################")

def plot_prices(import_price, storage_price, total_hours):
  largura = 0.35  # Largura das barras
  # Posições das barras
  posicoes = np.arange(total_hours)

  # Criando as barras
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.bar(posicoes - largura/2, storage_price, largura, label='Storage Energy', color='blue')
  ax.bar(posicoes + largura/2, import_price, largura, label='Import Energy', color='green')

  # Adicionando os labels e título
  ax.set_xlabel('Hours')
  ax.set_ylabel('Price Spent')
  ax.set_title('Energy prices per hour')
  ax.set_xticks(posicoes)  # Coloca as horas no eixo X
  ax.set_xticklabels([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,1])  # Exibe as horas de 0 a 23
  ax.legend()  # Exibe a legenda

  # Exibindo o gráfico
  plt.tight_layout()

  return plt.show()

def plot_accuracy(y_test, energy_supply):
  difference = []
  for i in range(len(energy_supply)):
      difference.append(y_test[i,0] + energy_supply[i])
  plt.figure(figsize=(10,6))
  # Ploting lines
  plt.plot(y_test[:len(energy_supply),0], label="Energy Consumption", color='blue', linestyle='dashed')
  plt.plot(energy_supply,label='Energy Supply', color = 'green', alpha = 0.7)
  plt.plot(difference, label="Rate", color='red', alpha=0.7)

  # Title and Label
  plt.title("Energy Rate with Storage Energy", fontsize=14)
  plt.xlabel("Hours", fontsize=12)
  plt.ylabel("Energetic Balance", fontsize=12)

  # Ading legends of lines
  plt.legend()

  # Show Graph
  return plt.show()

__All__ = [[options_filter, select_storage, calculate_cost ,plot_accuracy,plot_prices]]
