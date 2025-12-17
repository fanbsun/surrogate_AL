import numpy as np
import pickle, joblib
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats
from scipy.stats import spearmanr

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


def preprocess_inputdata_based_on_bins_positive(all_data):
  NUM_OF_BINS = 40
  input_data = []
  output = []
  errors = []
  z_data = []

  exlclude_paras = {"c": []}

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
    density_profiles.append(data['pos'][:NUM_OF_BINS,1])
    #density_profiles.append(data['neg'][:,1])
    output.append(density_profiles)
    density_errors.append(data['pos'][:NUM_OF_BINS,2])
    #density_errors.append(data['neg'][:,2])
    errors.append(density_errors)
    z_data_values.append(data['pos'][:NUM_OF_BINS,0])
    #z_data_values.append(data['neg'][:,0])
    z_data.append(z_data_values)

    #break

  input_data = np.array(input_data)
  output = np.array(output).reshape(-1,NUM_OF_BINS)
  errors = np.array(errors).reshape(-1,NUM_OF_BINS)
  z_data = np.array(z_data).reshape(-1,NUM_OF_BINS)
  print("Input data shape: {}".format(input_data.shape))
  print("Output data shape: {}".format(output.shape))
  print("error bar data shape: {}".format(errors.shape))
  print("Bin center data shape: {}".format(z_data.shape))

  return input_data, output, errors, z_data


def expand_input_output(input_data, output_data, n_positions=40):
    N, D = input_data.shape
    assert D == 7, f"Expected input_data shape (N, 7), got {input_data.shape}"
    assert output_data.shape == (N, n_positions), (
        f"Expected output_data shape ({N}, {n_positions}), got {output_data.shape}"
    )

    # Repeat each input sample for n_positions
    repeated_inputs = np.repeat(input_data, n_positions, axis=0)

    # Position vector 0..n_positions-1 repeated for each input
    #positions = np.tile(np.arange(n_positions), N).reshape(-1, 1)
    positions = np.arange(n_positions).reshape(-1, 1)
    positions_norm = positions / (n_positions - 1) 
    positions_full = np.tile(positions_norm, (N, 1))

    # Append position index to input
    expanded_input = np.hstack([repeated_inputs, positions_full])

    # Flatten output to match expanded input
    expanded_output = output_data.reshape(-1, 1)

    return expanded_input, expanded_output

def collapse_input_output(expanded_input, expanded_output, n_positions=40):
    assert expanded_input.ndim == 2 and expanded_input.shape[1] == 8, \
        f"expanded_input must be (N*n_positions, 8); got {expanded_input.shape}"
    assert expanded_output.ndim == 2 and expanded_output.shape[1] == 1, \
        f"expanded_output must be (N*n_positions, 1); got {expanded_output.shape}"

    total = expanded_input.shape[0]
    assert total % n_positions == 0, "Row count not divisible by n_positions."
    N = total // n_positions

    X7 = expanded_input[:, :7]
    pos_norm = expanded_input[:, 7]
    pos = np.rint(pos_norm * (n_positions - 1)).astype(int)
    y = expanded_output[:, 0]

    # reshape into blocks of size n_positions
    X7_blocks = X7.reshape(N, n_positions, 7)
    pos_blocks = pos.reshape(N, n_positions)
    y_blocks = y.reshape(N, n_positions)

    # sort each block by position, then verify X7 is constant within block
    input_data = np.empty((N, 7), dtype=X7.dtype)
    output_data = np.empty((N, n_positions), dtype=y_blocks.dtype)

    expected_pos = np.arange(n_positions)
    for i in range(N):
        order = np.argsort(pos_blocks[i])
        pos_sorted = pos_blocks[i, order]
        if not np.array_equal(pos_sorted, expected_pos):
            raise ValueError(
                f"Block {i}: position column is not a permutation of 0..{n_positions-1}."
            )

        X_sorted = X7_blocks[i, order, :]

        y_sorted = y_blocks[i, order]
        input_data[i] = X_sorted[0]
        output_data[i] = y_sorted

    return input_data, output_data


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

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_spearman(y_true, y_pred, std_dev):
    # Calculate absolute errors
    abs_errors = np.abs(y_true.flatten() - y_pred.flatten())
    
    # Calculate Spearman correlation between absolute errors and standard deviations
    correlation, _ = spearmanr(abs_errors, std_dev)
    return correlation


