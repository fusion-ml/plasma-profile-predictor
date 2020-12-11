from pathlib import Path
import sys
sys.path.append('..')
from helpers.data_generator import DataGenerator
from tqdm import trange
import pickle
import numpy as np
from ipdb import set_trace as db

def main(scenario_path, data_path, dump_path):
    with scenario_path.open('rb') as f:
        scenario = pickle.load(f, encoding='latin1')
    with data_path.open('rb') as f:
        data = pickle.load(f)
    scenario['process_data'] = False
    shuffle = False
    generator = DataGenerator(data,
                              1,
                              scenario['input_profile_names'],
                              scenario['actuator_names'],
                              scenario['target_profile_names'],
                              scenario['scalar_input_names'],
                              scenario['target_scalar_names'],
                              scenario['lookbacks'],
                              scenario['lookahead'],
                              scenario['predict_deltas'],
                              scenario['profile_downsample'],
                              shuffle,
                              sample_weights=scenario['sample_weighting'])
    bounds = {
            'a_EFIT01': (-1.8, 1.95),
            'betan_EFIT01': (-1.6, 1.6),
            'bt': (-0.34, 5.9),
            'curr': (-4.6, 1.4),
            'curr_target': (-1.3, 1.6),
            'dens': (-3, 3),
            'density_estimate': (-2, 2),
            # 'itemp': (-1, 4),
            'kappa_EFIT01': (-4, 2),
            'li_EFIT01': (-2, 3),
            'pinj': (-1.8, 2.5),
            'press_EFIT01': (-0.7, 3.7),
            'q_EFIT01': (-1.2, 2.5),
            'rotation': (-1, 3.5),
            'rmagx_EFIT01': (-2.3, 2),
            'target_density': (-1.2, 2.2),
            'temp': (-1, 2.4),
            'tinj': (-1.3, 1.7),
            'triangularity_bot_EFIT01': (-1.1, 1.3),
            'triangularity_top_EFIT01': (-1.7, 0.9),
            'volume_EFIT01': (-1.8, 1.4),
            }
    profile_inputs = generator.profile_inputs
    actuator_inputs = generator.actuator_inputs
    scalar_inputs = generator.scalar_inputs
    target_profiles = scenario['target_profile_names']
    target_scalars = scenario['target_scalar_names']
    lookahead = scenario['lookahead']
    normalization_dict = scenario['normalization_dict']
    profile_length = 33
    # action_space = spaces.Box(low=np.array([bounds[act][0] for act in actuator_inputs]),
                                   # high=np.array([bounds[act][1] for act in actuator_inputs]))
    obs_bot = []
    obs_top = []
    for sig in profile_inputs:
        obs_bot += [bounds[sig][0]] * profile_length
        obs_top += [bounds[sig][1]] * profile_length
    for sig in actuator_inputs + scalar_inputs:
        obs_bot += [bounds[sig][0]] * (scenario['lookbacks'][sig] + 1)
        obs_top += [bounds[sig][1]] * (scenario['lookbacks'][sig] + 1)
    obs_bot = np.array(obs_bot)
    obs_top = np.array(obs_top)
    action_bot = []
    action_top = []
    for sig in actuator_inputs:
        action_bot += [bounds[sig][0]]
        action_top += [bounds[sig][1]]
    action_bot = np.array(action_bot)
    action_top = np.array(action_top)

    data = []
    for i in trange(len(generator)):
        inp = generator[i][0]
        state = [inp['input_' + sig].flatten() for sig in profile_inputs] + \
                [inp['input_past_' + sig].flatten() for sig in actuator_inputs] + \
                [inp['input_' + sig].flatten() for sig in scalar_inputs]
        state = np.concatenate(state)
        state = np.clip(state, obs_bot, obs_top)
        action = [inp['input_future_' + sig][..., -1] for sig in actuator_inputs]
        action = np.concatenate(action)
        action = np.clip(action, action_bot, action_top)
        normalized_action = (action - action_bot) / (action_top - action_bot)
        data.append((state, normalized_action))
        # need to postprocess this for offline RL
    with dump_path.open('wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # val set
    plasma_path = Path('/zfsauton/project/public/virajm/plasma_models/')
    scenario_path = plasma_path / 'beta_n_included_params.pkl'
    data_path = plasma_path / 'val.pkl'
    out_path = Path('val_offline_data.pkl')
    print("starting val data")
    main(scenario_path, data_path, out_path)

    # train set
    plasma_path = Path('/zfsauton/project/public/virajm/plasma_models/')
    scenario_path = plasma_path / 'beta_n_included_params.pkl'
    data_path = plasma_path / 'train.pkl'
    out_path = Path('train_offline_data.pkl')
    print("starting train data")
    main(scenario_path, data_path, out_path)
