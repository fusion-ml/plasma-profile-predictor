import os
import gym
import keras
import pickle

from helpers.data_generator import DataGenerator


class ProfileEnv(gym.env):
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
        model_path = scenario_path[:-11] + '.h5'
        if not os.path.exists(model_path):
            raise ValueError(f"Path {model_path} doesn't exist!")
        self._model = keras.models.load_model(model_path, compile=False)
        print('loaded model: ' + model_path.split('/')[-1])
        self.state = None
        self.t = None
        self.t_max = 2200
        self.i = 0
        self.earliest_start_time = 200
        self.latest_start_time = 500
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
            example = self.val_generator[self.i]
            time = self.val_generator.cur_times[0]
            if time > self.earliest_start_time and time < self.latest_start_time:
                # TODO: arrange the data in state, return it
                pass
            i += 1
            if i > len(self.val_generator):
                i = 0


    def seed(self, seed=None):
        pass

    def step(self, action):
        states = self.make_states(self.state, action)
        self.state = self.predict(states)
        reward = self.compute_reward(self.state)
        self.t += 200
        done = self.t > self.t_max
        return self.state, reward, done, {}

    def compute_reward(self, state):
        return 0

    def predict(self, states):
        return self._model(states)

    def make_states(self, state, action):
        pass
