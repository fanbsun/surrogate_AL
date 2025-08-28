import os
import pickle
import numpy as np
import joblib
import argparse
from sklearn.utils import shuffle
from utils import preprocess_inputdata, compute_peak_density 


p = argparse.ArgumentParser()
p.add_argument("--pipeline_dir", required=True)
p.add_argument("--input_data_dir", required=True)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

#file_path= '/pscratch/sd/u/usatlas/globus-compute-test/Tianle_test/nano-confinement/surrogate_AL/data/'
with open(os.path.join(args.input_data_dir, 'data_dump_density_preprocessed_train.pk'), 'rb') as handle:
    processed_all_data_preprocessed_train = pickle.load(handle)
with open(os.path.join(args.input_data_dir, 'data_dump_density_preprocessed_test.pk'), 'rb') as handle:
    processed_all_data_preprocessed_test = pickle.load(handle)

#reduce training set size by randomly excluding N data.
np.random.seed(args.seed)   #fix random seed
index_ = np.random.choice(len(processed_all_data_preprocessed_train.keys()), 3500,replace=False)  #change choice size to select reduced training samples
excluded_index_ = np.delete(np.arange(0,len(processed_all_data_preprocessed_train.keys())), index_)
train_ = {}
exclude_ = {}
for index in index_:
    exclude_[list(processed_all_data_preprocessed_train.keys())[index]] = processed_all_data_preprocessed_train[list(processed_all_data_preprocessed_train.keys())[index]]
for index in excluded_index_:
    train_[list(processed_all_data_preprocessed_train.keys())[index]] = processed_all_data_preprocessed_train[list(processed_all_data_preprocessed_train.keys())[index]]


#Preprocess data to density output (NX1004)
input_data, output, errors, z_data = preprocess_inputdata(train_)
input_data_remain, output_remain, errors_remain, z_data_remain = preprocess_inputdata(exclude_)
input_data_test, output_test_raw, errors_test_raw, z_data_test = preprocess_inputdata(processed_all_data_preprocessed_test)


#Covert to peak density output (NX1)
output_train, errors_train = compute_peak_density(input_data, output, errors, z_data)
output_train_remain, errors_train_remain = compute_peak_density(input_data_remain, output_remain, errors_remain, z_data_remain)
output_test, errors_test = compute_peak_density(input_data_test, output_test_raw, errors_test_raw, z_data_test)


#cross validation 
#split ranges 0.8 to 1
train_test_split = 1


input_data_suff, output_suff,  errors_suff, z_data_shuff = shuffle(input_data, output_train, errors_train, z_data, random_state=args.seed)
#input_data_suff, output_suff,  errors_suff, z_data_shuff = shuffle(input_data, output[:, :100], errors[:, :100], z_data[:, :100], random_state=args.seed)

train_test_split_ = int(input_data_suff.shape[0]*train_test_split)

x_train = input_data_suff[0:train_test_split_]#.astype("float64")
x_test = input_data_suff[train_test_split_:]#.astype("float64")


y_train = output_suff[0:train_test_split_]#.astype("float64")
y_test = output_suff[train_test_split_:]#.astype("float64")


error_train = errors_suff[0:train_test_split_]#.astype("float64")
error_test = errors_suff[train_test_split_:]#.astype("float64")

z_data_train = z_data_shuff[0:train_test_split_]#.astype("float64")
z_data_test = z_data_shuff[train_test_split_:]#.astype("float64")


#x_train, x_test, y_train, y_test = spliter.train_test_split(input_data, output, test_size=(1-train_test_split), random_state=args.seed)

print("Train input: ", x_train.shape)
print("Train Output", y_train.shape)
print("Test input: ", x_test.shape)
print("Test Output", y_test.shape)



scaler = joblib.load(os.path.join(args.input_data_dir, 'scaler_new.pkl'))
scaled_x_train = scaler.transform(x_train)
scaled_x_test = scaler.transform(input_data_test)


scaler_y = joblib.load(os.path.join(args.input_data_dir, 'minmax_scaler_peak_label.joblib'))
scaler_error = joblib.load(os.path.join(args.input_data_dir, 'minmax_scaler_peak_error.joblib'))

scaled_y_train = scaler_y.transform(y_train)
scaled_y_test = scaler_y.transform(output_test)
scaled_error_train =  scaler_error.transform(error_train)
scaled_error_test =  scaler_error.transform(errors_test)


scaled_x_remain = scaler.transform(input_data_remain)
scaled_y_remain = scaler_y.transform(output_train_remain)
scaled_error_remain = scaler_error.transform(errors_train_remain)


out_dir = os.path.join(args.pipeline_dir, f"iter_001")
os.makedirs(out_dir, exist_ok=False)

np.save(os.path.join(out_dir, 'scaled_x_train.npy'),  scaled_x_train)
np.save(os.path.join(out_dir, 'scaled_y_train.npy'),  scaled_y_train)
np.save(os.path.join(out_dir, 'scaled_x_test.npy'),   scaled_x_test)
np.save(os.path.join(out_dir, 'scaled_y_test.npy'),   scaled_y_test)
np.save(os.path.join(out_dir, 'scaled_x_remain.npy'), scaled_x_remain)
np.save(os.path.join(out_dir, 'scaled_y_remain.npy'), scaled_y_remain)
