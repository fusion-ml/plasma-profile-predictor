import sys
import os
import pickle
import keras
import datetime
import matplotlib
import copy

import numpy as np
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import explained_variance_score
from ipdb import set_trace as db

sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, DataGenerator
from helpers.normalization import normalize, denormalize, renormalize
# from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
# from helpers.custom_losses import percent_correct_sign, baseline_MAE, normed_mse, mean_diff_sum_2, max_diff_sum_2, mean_diff2_sum2, max_diff2_sum2

##########
# set tf session
##########
config = tf.ConfigProto(intra_op_parallelism_threads=16,
                            inter_op_parallelism_threads=16,
                            allow_soft_placement=True,
                            device_count={'CPU': 8,
                                          'GPU': 1})
session = tf.Session(config=config)
K.set_session(session)


##########
# metrics
##########
def mean_squared_error(true,pred):
    return np.mean((true-pred)**2)

def mean_absolute_error(true,pred):
    return np.mean(np.abs(true-pred))

def median_absolute_error(true,pred):
    return np.median(np.abs(true-pred))

def percentile25_absolute_error(true,pred):
    return np.percentile(np.abs(true-pred),25)

def percentile75_absolute_error(true,pred):
    return np.percentile(np.abs(true-pred),75)

def median_squared_error(true,pred):
    return np.median((true-pred)**2)

def percentile25_squared_error(true,pred):
    return np.percentile((true-pred)**2,25)

def percentile75_squared_error(true,pred):
    return np.percentile((true-pred)**2,75)

metrics = {'mean_squared_error':mean_squared_error,
          'mean_absolute_error':mean_absolute_error,
          'median_absolute_error':median_absolute_error,
          'percentile25_absolute_error':percentile25_absolute_error,
          'percentile75_absolute_error':percentile75_absolute_error,
          'median_squared_error':median_squared_error,
          'percentile25_squared_error':percentile25_squared_error,
          'percentile75_squared_error':percentile75_squared_error}

##########
# load model and scenario
##########
base_path = '/zfsauton2/home/virajm/src/plasma-profile-predictor/outputs/'
# folders = ['run_results_no_parameters_no_stop/']
folders = ['beta_n_signals/']

for folder in folders:
    files =  [foo for foo in os.listdir(base_path+folder) if foo.endswith('.pkl')]
    for file in files:
        file_path = base_path + folder + file
        if not os.path.exists(file_path):
            print(f"Path {file_path} doesn't exist!")
            continue
        with open(file_path, 'rb') as f:
            scenario = pickle.load(f, encoding='latin1')
#         if 'evaluation_metrics' in scenario:
#             continue
        if len(scenario['target_profile_names']) > len(scenario['input_profile_names']):
            scenario['target_profile_names'] = scenario['input_profile_names']

        model_path = file_path[:-11] + '.h5'
        if not os.path.exists(model_path):
            print(f"Path {model_path} doesn't exist!")
            continue
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path, compile=False)
            print('loaded model: ' + model_path.split('/')[-1])
        else:
            print('no model for path:',model_path)
            continue
        if 'target_scalar_names' not in scenario:
            scenario['target_scalar_names'] = []

        full_data_path = '/zfsauton2/home/virajm/data/profile_data/train_data_full.pkl'
        # rt_data_path = '/scratch/gpfs/jabbate/test_rt/final_data.pkl'
        traindata, valdata, normalization_dict = helpers.data_generator.process_data(full_data_path,
                                                          scenario['sig_names'],
                                                          scenario['normalization_method'],
                                                          scenario['window_length'],
                                                          scenario['window_overlap'],
                                                          scenario['lookbacks'],
                                                          scenario['lookahead'],
                                                          scenario['sample_step'],
                                                          scenario['uniform_normalization'],
                                                          scenario['train_frac'],
                                                          scenario['val_frac'],
                                                          scenario['nshots'],
                                                          0, #scenario['verbose']
                                                          scenario['flattop_only'],
                                                          randomize=False,
                                                          pruning_functions=scenario['pruning_functions'],
                                                          excluded_shots = scenario['excluded_shots'],
                                                          delta_sigs = [],
                                                          invert_q=scenario.setdefault('invert_q',False),
                                                          val_idx = scenario['val_idx'])
        valdata = helpers.normalization.renormalize(
            helpers.normalization.denormalize(
                valdata.copy(),normalization_dict, verbose=0),
            scenario['normalization_dict'],verbose=0)
        traindata = helpers.normalization.renormalize(
            helpers.normalization.denormalize(
                traindata.copy(),normalization_dict, verbose=0),
            scenario['normalization_dict'],verbose=0)

        db()

        train_generator = DataGenerator(traindata,
                                        scenario['batch_size'],
                                        scenario['input_profile_names'],
                                        scenario['actuator_names'],
                                        scenario['target_profile_names'],
                                        scenario['scalar_input_names'],
                                        scenario['target_scalar_names'],
                                        scenario['lookbacks'],
                                        scenario['lookahead'],
                                        scenario['predict_deltas'],
                                        scenario['profile_downsample'],
                                        False,
                                        sample_weights = None)

#         optimizer = keras.optimizers.Adam()
#         loss = keras.metrics.mean_squared_error
#         metrics = [keras.metrics.mean_squared_error, 
#                    keras.metrics.mean_absolute_error, 
#                    normed_mse, 
#                    mean_diff_sum_2, 
#                    max_diff_sum_2, 
#                    mean_diff2_sum2, 
#                    max_diff2_sum2]
#         model.compile(optimizer, loss, metrics)

#         outs = model.evaluate_generator(train_generator, verbose=0, workers=4, use_multiprocessing=True)
        targets = scenario['target_profile_names'].copy()
        targets += scenario['target_scalar_names']

        predictions_arr = model.predict_generator(train_generator, verbose=0)# , workers=4, use_multiprocessing=True)

        predictions = {sig: arr for sig, arr in zip(targets, predictions_arr)}


        baseline = {sig:[] for sig in targets}
        for i in range(len(train_generator)):
            sample = train_generator[i]
            for sig in targets:
                baseline[sig].append(sample[1]['target_'+sig])
        baseline = {sig:np.concatenate(baseline[sig],axis=0) for sig in targets}
        if len(scenario['target_profile_names']) > 0:
            # get profile data in numpy arrays
            Y_hat_profiles = np.concatenate([predictions[key] for key in scenario['target_profile_names']], axis=1)
            Y_profiles = np.concatenate([baseline[key].reshape(predictions[key].shape) for key in scenario['target_profile_names']], axis=1)

            explained_variance_profiles = explained_variance_score(Y_profiles, Y_hat_profiles)
            print(f"Explained variance on profiles: {explained_variance_profiles:.2%}")

        if len(scenario['target_scalar_names']) > 0:
            # get scalar data in numpy arrays
            Y_hat_scalars = np.concatenate([predictions[key] for key in scenario['target_scalar_names']], axis=1)
            Y_scalars = np.concatenate([baseline[key].reshape(predictions[key].shape) for key in scenario['target_scalar_names']], axis=1)

            explained_variance_scalars = explained_variance_score(Y_scalars, Y_hat_scalars)
            print(f"Explained variance on scalars: {explained_variance_scalars:.2%}")

            explained_variance_scalars = explained_variance_score(Y_scalars, Y_hat_scalars, multioutput='raw_values')
            print(f"Broken down:")
            for i, name in enumerate(scenario['target_scalar_names']):
                print(f"{name}: {explained_variance_scalars[i]:.2%}")

        evaluation_metrics = {}
        for metric_name,metric in metrics.items():
            s = 0
            for sig in targets:
                key = sig + '_' + metric_name
                val = metric(baseline[sig],predictions[sig])
                s += val/len(targets)
                evaluation_metrics[key] = val
                print(key)
                print(val)
            evaluation_metrics[metric_name] = s

        scenario['evaluation_metrics'] = evaluation_metrics
        if 'date' not in scenario:
            scenario['date'] = datetime.datetime.strptime(scenario['runname'].split('_')[-2],'%d%b%y-%H-%M')

        with open(file_path,'wb+') as f:
            pickle.dump(copy.deepcopy(scenario),f)

        print('saved evaluation metrics')
        print(evaluation_metrics)

    print('done')
