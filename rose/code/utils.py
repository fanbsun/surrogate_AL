import numpy as np
import pickle, joblib
from sklearn.metrics import mean_squared_error

def preprocess_inputdata(all_data):
  NUM_OF_BINS = 502
  input_data = []
  output = []
  errors = []
  z_data = []

  #exlclude_paras = {"c": ["0.25", "0.75", "1.25", "1.75"]}
  exlclude_paras = {}
  for key, data in all_data.items():
    #print(key, data)
    density_profiles = []
    density_errors = []
    z_data_values = []
    input_names = key.split("_")[0::2]
    input_paras = key.split("_")[1::2]

    ignore_this = False
    for key_p, params in exlclude_paras.items():
        if input_paras[input_names.index(key_p)] in params:
            ignore_this= True
            break
    if ignore_this:
        continue

    input_data.append(input_paras)
    density_profiles.append(data['pos'][:,1])
    density_profiles.append(data['neg'][:,1])
    output.append(density_profiles)
    density_errors.append(data['pos'][:,2])
    density_errors.append(data['neg'][:,2])
    errors.append(density_errors)
    z_data_values.append(data['pos'][:,0])
    z_data_values.append(data['neg'][:,0])
    z_data.append(z_data_values)

    #break

  input_data = np.array(input_data)
  output = np.array(output).reshape(-1,NUM_OF_BINS*2)
  errors = np.array(errors).reshape(-1,NUM_OF_BINS*2)
  z_data = np.array(z_data).reshape(-1,NUM_OF_BINS*2)
  print("Input data shape: {}".format(input_data.shape))
  print("Output data shape: {}".format(output.shape))
  print("error bar data shape: {}".format(errors.shape))
  print("Bin center data shape: {}".format(z_data.shape))

  return input_data, output, errors, z_data

def compute_peak_density(input_data, output, errors, z_data):
    output_peak_density = np.zeros((input_data.shape[0], 1))
    error_peak_density = np.zeros((input_data.shape[0], 1))
    for i in range(input_data.shape[0]):
        max_index = np.argmax(output[i])
        output_peak_density[i] = output[i][max_index]
        error_peak_density[i] = errors[i][max_index]
    return output_peak_density, error_peak_density

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
