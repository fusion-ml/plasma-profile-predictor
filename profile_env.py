import os
from gym import Env, spaces
import keras
import pickle

from helpers.data_generator import DataGenerator
from ipdb import set_trace as db


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
        self.profile_inputs = self.val_generator.profile_inputs
        self.actuator_inputs = self.val_generator.actuator_inputs
        self.scalar_inputs = self.val_generator.scalar_inputs
        self.action_spaces = spaces.Box(low=-2, high=2, shape=(len(actuator_inputs),))
        self.observation_space = spaces.Box(low=-2
        model_path = scenario_path[:-11] + '.h5'
        if not os.path.exists(model_path):
            raise ValueError(f"Path {model_path} doesn't exist!")
        self._model = keras.models.load_model(model_path, compile=False)
        self._state = None
        print('loaded model: ' + model_path.split('/')[-1])
        self._state = None
        self.t = None
        self.timestep
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
                self._state = example
                return self.state
            i += 1
            if i > len(self.val_generator):
                i = 0

    @property
    def state(self):
        return [self._state['input_' + sig] for sig in self.profile_inputs] + \
               [self._state['input_past_' + sig] for sig in self.actuator_inputs] + \
               # don't use future actuators because they are the action
               # [self._state['input_future_' + sig] for sig in self.actuator_inputs] + \
               [inputs['input_' + sig] for sig in self.scalar_inputs]

    def seed(self, seed=None):
        pass

    def step(self, action):
        states = self.make_states(self.state, action)
        self.state = self.predict(states)
        reward = self.compute_reward(self.state)
        self.t += self.timestep
        done = self.t > self.t_max
        return self.state, reward, done, {}

    def compute_reward(self, state):
        return 0

    def predict(self, states):
        return self._model.predict(states)

    def make_states(self, state, actions):
        n_actions = actions.shape[0]
        states = {}
        for name, array in state.items():
            repeated_array = np.tile(array, (n_actions, 1))
            states[name] = repeated_array
        for i in range(actions.shape[1]):
            name = self.actuator_inputs[i]
            states['input_future_' + name] = actions[:, i:i+1]
        return states



def test_env():
    env = ProfileEnv(scenario_path="/zfsauton2/home/virajm/src/plasma-profile-predictor/outputs/parameters_no_stop_joe_thicc/model-conv2d_profiles-dens-temp-itemp-q_EFIT01-rotation_act-target_density-pinj-tinj-curr_target_22Jul20-08-51_params.pkl")
    env.reset()

if __name__ == '__main__':
    test_env()
