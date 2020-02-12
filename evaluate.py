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

sys.path.append(os.path.abspath('../'))
import helpers
from helpers.data_generator import process_data, DataGenerator
from helpers.normalization import normalize, denormalize, renormalize
from helpers.custom_losses import denorm_loss, hinge_mse_loss, percent_baseline_error, baseline_MAE
from helpers.custom_losses import percent_correct_sign, baseline_MAE, normed_mse, mean_diff_sum_2, max_diff_sum_2, mean_diff2_sum2, max_diff2_sum2

##########
# set tf session
##########
config = tf.ConfigProto(intra_op_parallelism_threads=32,
                            inter_op_parallelism_threads=32,
                            allow_soft_placement=True,
                            device_count={'CPU': 16,
                                          'GPU': 0})
session = tf.Session(config=config)
K.set_session(session)


##########
# load model and scenario
##########

base_path = os.path.expanduser('~/run_results_01_31/')
files = [foo for foo in os.listdir(base_path) if foo.endswith('.h5')]
all_eval = {}
for file in files:
    file_path = base_path + file
    model = keras.models.load_model(file_path, compile=False)
    print('loaded model: ' + file_path.split('/')[-1])
    file_path = file_path[:-3] + '_params.pkl'
    with open(file_path, 'rb') as f:
        scenario = pickle.load(f, encoding='latin1')
    print('loaded dict: ' + file_path.split('/')[-1])
    print('with parameters: ' + str(scenario.keys()))
    
    full_data_oath = '/scratch/gpfs/jabbate/full_data/train_data_full.pkl'
    test_data_path = '/scratch/gpfs/jabbate/full_data/test_data.pkl' 
    traindata, valdata, normalization_dict = helpers.data_generator.process_data(test_data_path,
                                                      scenario['sig_names'],
                                                      scenario['normalization_method'],
                                                      scenario['window_length'],
                                                      scenario['window_overlap'],
                                                      scenario['lookbacks'],
                                                      scenario['lookahead'],
                                                      scenario['sample_step'],
                                                      scenario['uniform_normalization'],
                                                      1, #scenario['train_frac'],
                                                      0, #scenario['val_frac'],
                                                      scenario['nshots'],
                                                      0, #scenario['verbose']
                                                      scenario['flattop_only'],
                                                      randomize=False,
                                                      pruning_functions=scenario['pruning_functions'],
                                                      excluded_shots = scenario['excluded_shots'],
                                                      delta_sigs = [])
    traindata = helpers.normalization.renormalize(
        helpers.normalization.denormalize(
            traindata.copy(),normalization_dict, verbose=0),
        scenario['normalization_dict'],verbose=0)
    
    train_generator = DataGenerator(traindata,
                                    scenario['batch_size'],
                                    scenario['input_profile_names'],
                                    scenario['actuator_names'],
                                    scenario['target_profile_names'],
                                    scenario['scalar_input_names'],
                                    scenario['lookbacks'],
                                    scenario['lookahead'],
                                    scenario['predict_deltas'],
                                    scenario['profile_downsample'],
                                    False,
                                    sample_weights = None)
    
    optimizer = keras.optimizers.Adam()
    loss = keras.metrics.mean_squared_error
    metrics = [keras.metrics.mean_squared_error, 
               keras.metrics.mean_absolute_error, 
               normed_mse, 
               mean_diff_sum_2, 
               max_diff_sum_2, 
               mean_diff2_sum2, 
               max_diff2_sum2]
    model.compile(optimizer, loss, metrics)

    outs = model.evaluate_generator(train_generator, verbose=1, workers=8, use_multiprocessing=True)
    
    
    evaluation_metrics = {name: val for name,val in zip(model.metrics_names,outs)}
    for metric in metrics:
        name = metric if isinstance(metric,str) else str(metric.__name__)
        print(name)
        s = 0
        for key,val in evaluation_metrics.items():
            if name in key:
                s += val/len(model.outputs)
        print(s)
        evaluation_metrics[name] = s
    
    
    scenario['regular_evaluation_metrics'] = evaluation_metrics
    if 'date' not in scenario:
        scenario['date'] = datetime.datetime.strptime(scenario['runname'].split('_')[-2],'%d%b%y-%H-%M')
        
    with open(file_path,'wb+') as f:
        pickle.dump(scenario,f)
        
    print('saved evaluation metrics')
    print(evaluation_metrics)
    
    all_eval[scenario['runname']] = evaluation_metrics
    
with open(base_path + 'evaluation_parameters.pkl','wb+') as f:
        pickle.dump(all_eval,f)
        
print('done')