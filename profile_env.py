import os
from gym import Env, spaces
import keras
import pickle
import numpy as np

from helpers.data_generator import DataGenerator


class ProfileEnv(Env):
    def __init__(self, scenario_path):
        if not os.path.exists(scenario_path):
            raise ValueError(f"Scenario Path {scenario_path} does not exist!")
        with open(scenario_path, 'rb') as f:
            self.scenario = pickle.load(f, encoding='latin1')
        self.scenario['process_data'] = False
        with open(os.path.join(self.scenario['processed_filename_base'], 'val.pkl'), 'rb') as f:
            valdata = pickle.load(f)
        self.val_generator = DataGenerator(valdata,
                                           1,
                                           self.scenario['input_profile_names'],
                                           self.scenario['actuator_names'],
                                           self.scenario['target_profile_names'],
                                           self.scenario['scalar_input_names'],
                                           self.scenario['target_scalar_names'],
                                           self.scenario['lookbacks'],
                                           self.scenario['lookahead'],
                                           self.scenario['predict_deltas'],
                                           self.scenario['profile_downsample'],
                                           self.scenario['shuffle_generators'],
                                           sample_weights=self.scenario['sample_weighting'])
        self.time_lookback = self.scenario['lookbacks']['time']
        self.bounds = {
                'curr_target': (-1.3, 1.6),
                'dens': (-3, 3),
                'density_estimate': (-2, 2),
                'itemp': (-1, 4),
                'kappa_EFIT01': (-4, 2),
                'li_EFIT01': (-2, 3),
                'pinj': (-1.8, 2.5),
                'q_EFIT01': (-1.2, 2.5),
                'rotation': (-1, 3.5),
                'target_density': (-1.2, 2.2),
                'temp': (-1, 2.4),
                'tinj': (-1.3, 1.7),
                'triangularity_bot_EFIT01': (-1.1, 1.3),
                'triangularity_top_EFIT01': (-1.7, 0.9),
                'volume_EFIT01': (-1.8, 1.4),
                }

        self.profile_inputs = self.val_generator.profile_inputs
        self.actuator_inputs = self.val_generator.actuator_inputs
        self.scalar_inputs = self.val_generator.scalar_inputs
        self.target_profiles = self.scenario['target_profile_names']
        self.target_scalars = self.scenario['target_scalar_names']
        self.lookahead = self.scenario['lookahead']
        self.profile_length = 33
        self.action_space = spaces.Box(low=np.array([self.bounds[act][0] for act in self.actuator_inputs]),
                                        high=np.array([self.bounds[act][1] for act in self.actuator_inputs]))
        obs_bot = []
        obs_top = []
        for sig in self.profile_inputs:
            obs_bot += [self.bounds[sig][0]] * self.profile_length
            obs_top += [self.bounds[sig][1]] * self.profile_length
        for sig in self.actuator_inputs + self.scalar_inputs:
            obs_bot += [self.bounds[sig][0]] * (self.scenario['lookbacks'][sig] + 1)
            obs_top += [self.bounds[sig][1]] * (self.scenario['lookbacks'][sig] + 1)
        obs_bot = np.array(obs_bot)
        obs_top = np.array(obs_top)

        self.observation_space = spaces.Box(low=obs_bot, high=obs_top)

        model_path = scenario_path[:-11] + '.h5'
        if not os.path.exists(model_path):
            raise ValueError(f"Path {model_path} doesn't exist!")
        self._model = keras.models.load_model(model_path, compile=False)
        self._state = None
        print('loaded model: ' + model_path.split('/')[-1])
        self._state = None
        self.t = None
        self.timestep = 50
        self.t_max = 2200
        self.i = 0
        self.earliest_start_time = 500
        self.latest_start_time = 1000
        '''
        self.validation_data = [inputs['input_' + sig] for sig in self.profile_inputs] + \
                               [inputs['input_past_' + sig] for sig in self.actuator_inputs] + \
                               [inputs['input_future_' + sig] for sig in self.actuator_inputs] + \
                               [inputs['input_' + sig] for sig in self.scalar_inputs] + \
                               [targets['target_' + sig] if len(targets['target_' + sig].shape) == 2 else targets['target_' + sig][:, np.newaxis]
                                   for sig in self.target_names] + [self.sample_weights for _ in range(len(self.target_names))]
        '''

    def reset(self):
        while True:
            example = self.val_generator[self.i][0]
            time = self.val_generator.cur_times[0, self.time_lookback]
            if time > self.earliest_start_time and time < self.latest_start_time:
                # TODO: arrange the data in state, return it
                self._state = example
                self.t = 0
                return self.state
            self.i += 1
            if self.i == len(self.val_generator):
                print("Went through whole generator, restarting.")
                self.i = 0

    @property
    def state(self):
        state = [self._state['input_' + sig].flatten() for sig in self.profile_inputs] + \
                [self._state['input_past_' + sig].flatten() for sig in self.actuator_inputs] + \
                [self._state['input_' + sig].flatten() for sig in self.scalar_inputs]
        return np.array(state)
                # don't use future actuators because they are the action
                # [self._state['input_future_' + sig] for sig in self.actuator_inputs] + \

    def seed(self, seed=None):
        pass

    def set_state(self, states, action, output):
        new_state = {}
        for i, prof in enumerate(self.target_profiles):
            new_state['input_' + prof] = states['input_' + prof][0, ...] + output[i][0, ...]

        for act in self.actuator_inputs:
            new_state['input_past_' + act] = np.concatenate((states['input_past_' + act][0, -3:], states['input_future_' + act][0, :]))

        for i, scalar in enumerate(self.scalar_inputs):
            i += len(self.target_profiles)
            last_scalar = states['input_' + scalar][0, -1]
            new_last_scalar = last_scalar + output[i][0, 0]
            theta = ((np.arange(self.lookahead) + 1) / self.lookahead)
            interpolated_scalar = theta * new_last_scalar + (1 - theta) * last_scalar
            new_state['input_' + scalar] = np.concatenate((states['input_' + scalar][0, -3:], interpolated_scalar))
        self._state = new_state




    def step(self, action):
        states = self.make_states(self._state, action)
        output = self.predict(states)
        self.set_state(states, action, output)
        # self._state = dict(zip(self.target_profiles + self.target_scalars, output))
        reward = self.compute_reward(self.state)
        self.t += self.timestep
        done = self.t > self.t_max
        return self.state, reward, done, {}

    def compute_reward(self, state):
        return 0

    def predict(self, states):
        return self._model.predict(states)

    def make_states(self, state, actions):
        if actions.ndim == 1:
            actions = actions[np.newaxis, ...]
        n_actions = actions.shape[0]
        states = {}
        for name, array in state.items():
            repeated_array = np.tile(array, (n_actions, 1))
            if repeated_array.shape[-1] == self.profile_length and repeated_array.ndim == 2:
                repeated_array = repeated_array[:, np.newaxis, :]
            states[name] = repeated_array
        actions = self.interpolate_actions(states, actions)
        states.update(actions)
        return states

    def interpolate_actions(self, states, actions):
        new_actions = {}
        for i, sig in enumerate(self.actuator_inputs):
            old_action = states['input_past_' + sig][0, -1]
            new_action = actions[:, i]
            theta = ((np.arange(self.lookahead) + 1) / self.lookahead)[np.newaxis, ...]
            interpolated_action = theta * new_action + (1 - theta) * old_action
            new_actions['input_future_' + sig] = interpolated_action
        return new_actions




def test_env():
    env = ProfileEnv(scenario_path="/zfsauton2/home/virajm/src/plasma-profile-predictor/outputs/parameters_no_stop_joe_thicc/model-conv2d_profiles-dens-temp-itemp-q_EFIT01-rotation_act-target_density-pinj-tinj-curr_target_22Jul20-08-51_params.pkl")
    env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        if done:
            break

if __name__ == '__main__':
    test_env()
