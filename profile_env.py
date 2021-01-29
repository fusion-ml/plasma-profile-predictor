import os
from gym import Env, spaces
import keras
import pickle
import numpy as np
from pathlib import Path
from tqdm import trange
import tensorflow as tf

from helpers.data_generator import DataGenerator
from helpers.normalization import denormalize
from utils import get_historical_slice
from nn_tearing_wrapper import NNTearingModel
from stability.disrupt_predictor import load_cb_from_files
from ipdb import set_trace as db


# SCENARIO_PATH = "/zfsauton2/home/virajm/src/plasma-profile-predictor/outputs/beta_n_signals/model-conv2d_profiles-dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_30Jul20-16-13_params.pkl"  # NOQA
SCENARIO_PATH = '/zfsauton/project/public/virajm/plasma_models/test_abs_recent_params.pkl'
TEARING_PATH = Path('/zfsauton/project/public/ichar/FusionModels/tearing')
NN_TEARING_PATH = Path('/zfsauton/project/public/ichar/FusionModels/nn_tearing')
VAL_PATH = Path('/zfsauton/project/public/virajm/plasma_models/val.pkl')
TRAIN_PATH = Path('/zfsauton/project/public/virajm/plasma_models/train.pkl')

SHUFFLE_STARTS = False


class ProfileEnv(Env):
    def __init__(self, scenario_path, gpu_num=None):
        if not os.path.exists(scenario_path):
            raise ValueError(f"Scenario Path {scenario_path} does not exist!")
        with open(scenario_path, 'rb') as f:
            self.scenario = pickle.load(f, encoding='latin1')
        self.scenario['process_data'] = False
        with VAL_PATH.open('rb') as f:
            valdata = pickle.load(f)
        with TRAIN_PATH.open('rb') as f:
            traindata = pickle.load(f)
        shuffle = SHUFFLE_STARTS and self.scenario['shuffle_generators']
        self.train_generator = DataGenerator(traindata,
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
                                           shuffle,
                                           sample_weights=self.scenario['sample_weighting'])
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
                                           shuffle,
                                           sample_weights=self.scenario['sample_weighting'])
        self.time_lookback = self.scenario['lookbacks']['time']
        self.target_beta_n = 1.5
        self.bounds = {
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
        self.profile_inputs = self.val_generator.profile_inputs
        self.actuator_inputs = self.val_generator.actuator_inputs
        self.scalar_inputs = self.val_generator.scalar_inputs
        self.target_profiles = self.scenario['target_profile_names']
        self.target_scalars = self.scenario['target_scalar_names']
        self.lookahead = self.scenario['lookahead']
        self.normalization_dict = self.scenario['normalization_dict']
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
        if gpu_num:
            with tf.device(f"gpu:{gpu_num}"):
                self._model = keras.models.load_model(model_path, compile=False)
        else:
            self._model = keras.models.load_model(model_path, compile=False)
        self._state = None
        self._state = None
        self.t = None
        self.absolute_time = None
        self.timestep = 200  # ms
        self.tau = 0.2  # seconds
        self.t_max = 5000
        self.flattop_dcurr_max = 10000 # A / ms
        self.i = 0
        self.earliest_start_time = 1100
        self.latest_start_time = 1600
        self.mu_0 = 1.256637E-6
        self.current_beta_n = None
        self.current_field_strength = None
        self.current_plasma_pressure = None
        self.current_beta = None
        self.current_minor_radius = None
        self.current_current = None
        self.eps_denominator = 1e-4
        '''
        self.validation_data = [inputs['input_' + sig] for sig in self.profile_inputs] + \
                               [inputs['input_past_' + sig] for sig in self.actuator_inputs] + \
                               [inputs['input_future_' + sig] for sig in self.actuator_inputs] + \
                               [inputs['input_' + sig] for sig in self.scalar_inputs] + \
                               [targets['target_' + sig] if len(targets['target_' + sig].shape) == 2 else targets['target_' + sig][:, np.newaxis]
                                   for sig in self.target_names] + [self.sample_weights for _ in range(len(self.target_names))]
        '''

    def reset(self):
        self.current_beta_n = None
        self.current_field_strength = None
        self.current_plasma_pressure = None
        self.current_beta = None
        self.current_minor_radius = None
        self.current_current = None
        while True:
            example = self.val_generator[self.i][0]
            time = self.val_generator.cur_times[0, self.time_lookback]
            denorm_example = denormalize(example, self.normalization_dict, verbose=False)
            curr = denorm_example['input_curr']
            curr_derivative_approx = (curr[0, -1] - curr[0, 0]) / (len(curr) * 50)  # should be in A / ms
            self.i += 1
            if np.abs(curr_derivative_approx) > self.flattop_dcurr_max:
                continue
            if time > self.earliest_start_time and time < self.latest_start_time:
                self._state = example
                self.t = 0
                self.absolute_time = time
                self.i += 1
                return self.obs
            if self.i == len(self.val_generator):
                print("Went through whole generator, restarting.")
                self.i = 0

    @property
    def obs(self):
        state = [self._state['input_' + sig].flatten() for sig in self.profile_inputs] + \
                [self._state['input_past_' + sig].flatten() for sig in self.actuator_inputs] + \
                [self._state['input_' + sig].flatten() for sig in self.scalar_inputs]
        # don't use future actuators because they are the action
        # [self._state['input_future_' + sig] for sig in self.actuator_inputs] + \
        return np.concatenate(state)

    def state_to_obs(self, state):
        state = [state['input_' + sig].reshape((-1, self.profile_length)) for sig in self.profile_inputs] + \
                [state['input_past_' + sig].reshape((-1, self.time_lookback + 1)) for sig in self.actuator_inputs] + \
                [state['input_' + sig].reshape((-1, self.time_lookback + 1)) for sig in self.scalar_inputs]
        return np.concatenate(state, axis=1)

    def get_value_from_denorm_state(self, denorm_state, name):
        return denorm_state['input_' + name]

    def obs_to_state(self, obs):
        # obs is a vector, state is a dict
        state = {}
        for i, sig in enumerate(self.profile_inputs):
            state['input_' + sig] = obs[..., i * self.profile_length:(i + 1) * self.profile_length]
        total_profile_inputs = len(self.profile_inputs) * self.profile_length
        scalar_timesteps = self.time_lookback + 1
        for i, sig in enumerate(self.actuator_inputs):
            state['input_past_' + sig] = obs[..., total_profile_inputs + (scalar_timesteps * i):total_profile_inputs +
                                             (scalar_timesteps * (i + 1))]
        total_prev_inputs = total_profile_inputs + scalar_timesteps * len(self.actuator_inputs)
        for i, sig in enumerate(self.scalar_inputs):
            state['input_' + sig] = obs[..., total_prev_inputs + (scalar_timesteps * i):total_prev_inputs +
                                        (scalar_timesteps * (i + 1))]
        return state

    def seed(self, seed=None):
        pass

    def output_to_state(self, states, action, output):
        new_state = {}
        for i, prof in enumerate(self.target_profiles):
            baseline = states['input_' + prof][:, 0, :]
            new_state['input_' + prof] = baseline + output[i]

        for act in self.actuator_inputs:
            new_state['input_past_' + act] = np.concatenate((states['input_past_' + act][:, -3:],
                                                             states['input_future_' + act]), axis=1)

        for i, scalar in enumerate(self.scalar_inputs):
            i += len(self.target_profiles)
            last_scalar = states['input_' + scalar][:, -1:]
            new_last_scalar = last_scalar + output[i][:, :1]
            theta = ((np.arange(self.lookahead) + 1) / self.lookahead)[np.newaxis, ...]
            interpolated_scalar = theta * new_last_scalar + (1 - theta) * last_scalar
            if interpolated_scalar.ndim == 1:
                interpolated_scalar = interpolated_scalar[np.newaxis, ...]
            new_state['input_' + scalar] = np.concatenate((states['input_' + scalar][:, -3:], interpolated_scalar),
                                                          axis=1)
        return new_state

    def step(self, action):
        states = self.make_states(self._state, action)
        output = self.predict(states)
        self._state = self.output_to_state(states, action, output)
        # self._state = dict(zip(self.target_profiles + self.target_scalars, output))
        reward = self.compute_reward(self._state)
        self.t += self.timestep
        self.absolute_time += self.timestep
        done = self.t > self.t_max
        info = {
                'beta_n': self.current_beta_n,
                'field_strength': self.current_field_strength,
                'plasma_pressure': self.current_plasma_pressure,
                'beta': self.current_beta,
                'minor_radius': self.current_minor_radius,
                'current': self.current_current
               }
        return self.obs, reward, done, info

    def _compute_beta_n(self, state):
        pressure_profile = state['input_press_EFIT01']  # Pa
        mean_total_field_strength = np.abs(state['input_bt'][..., -1])
        # Here we're making the assumption that B ~= B_t as
        # most of the magnetic field is composed of the
        # toroidal component. Also denoted in Tesla.
        mean_total_field_strength = np.maximum(mean_total_field_strength, self.eps_denominator)
        self.current_field_strength = mean_total_field_strength
        mean_plasma_pressure = np.mean(pressure_profile, axis=-1)  # TODO: take the geometry of the torus into account
        mean_plasma_pressure = np.maximum(mean_plasma_pressure, 0)
        self.current_plasma_pressure = mean_plasma_pressure
        beta = mean_plasma_pressure * 2 * self.mu_0 / mean_total_field_strength ** 2
        self.current_beta = beta
        minor_radius = state['input_a_EFIT01'][..., -1]  # meters
        minor_radius = np.maximum(minor_radius, 0)
        self.current_minor_radius = minor_radius
        current = np.abs(state['input_curr'][..., -1] / 1e6)  # convert to MA from amps
        current = np.maximum(current, self.eps_denominator)  # use eps for numerical stability
        self.current_current = current
        beta_n = beta * minor_radius * mean_total_field_strength / current
        return beta_n * 100  # have to convert beta_n to a percent

    def compute_beta_n(self, obs):
        state = self.obs_to_state(obs)
        denorm_state = denormalize(state, self.normalization_dict, verbose=False)
        return self._compute_beta_n(denorm_state)


    def unroll(self, obs, action_sequence):
        """
        obs: batch_size * obs_dim (ndarray)
        action_sequence: batch_size * timesteps * action_dim
        """
        batch_size = action_sequence.shape[0]
        n_timesteps = action_sequence.shape[1]
        obs = np.tile(obs, (batch_size, 1))
        obs_sequence = []
        rew_sequence = []
        state = self.obs_to_state(obs)
        for i in range(n_timesteps):
            action = action_sequence[:, i, :]
            model_input = self.make_states(state, action)
            obs = self.predict(model_input)
            state = self.output_to_state(model_input, action, obs)
            rewards = self.compute_reward(state)
            obs_sequence.append(self.state_to_obs(state))
            rew_sequence.append(rewards)
        obs_sequence, rew_sequence = np.stack(obs_sequence), np.stack(rew_sequence)
        return np.transpose(obs_sequence, (1, 0, 2)), np.transpose(rew_sequence, (1, 0))

    def compute_reward(self, state):
        denorm_state = denormalize(state, self.normalization_dict, verbose=False)
        self.current_beta_n = self._compute_beta_n(denorm_state)
        return -(self.current_beta_n - self.target_beta_n) ** 2

    def predict(self, states):
        return self._model.predict(states)

    def make_states(self, state, actions):
        if actions.ndim == 1:
            actions = actions[np.newaxis, ...]
        states = {}
        for name, array in state.items():
            if array.ndim == 1:
                array = array[np.newaxis, ...]
            # repeated_array = array
            if array.shape[-1] == self.profile_length and array.ndim == 2:
                array = array[:, np.newaxis, :]
            states[name] = array
        actions = self.interpolate_actions(states, actions)
        states.update(actions)
        return states

    def interpolate_actions(self, states, actions):
        new_actions = {}
        for i, sig in enumerate(self.actuator_inputs):
            old_action = states['input_past_' + sig][:, -1:]
            new_action = actions[:, i:i+1]
            theta = ((np.arange(self.lookahead) + 1) / self.lookahead)[np.newaxis, ...]
            interpolated_action = theta * new_action + (1 - theta) * old_action
            new_actions['input_future_' + sig] = interpolated_action
        return new_actions


class TearingProfileEnv(ProfileEnv):
    def __init__(self, scenario_path, tearing_path, rew_coefs, gpu_num=None,
                 nn_tearing=False):
        super().__init__(scenario_path, gpu_num)
        self.tearing_path = tearing_path
        self.current_tearing_prob = None
        self.rew_coefs = rew_coefs
        self.tearing_headers = [('input_kappa_EFIT01', 'kappa'),
                                ('input_triangularity_top_EFIT01', 'tritop'),
                                ('input_triangularity_bot_EFIT01', 'tribot'),
                                ('input_rmagx_EFIT01', 'R0'),
                                ('input_volume_EFIT01', 'efsvolume'),
                                ('input_a_EFIT01', 'aminor'),
                                ('input_density_estimate', 'dssdenest'),
                                ('input_curr', 'ip'),
                                ('input_li_EFIT01', 'efsli')]
        self.tearing_history_window = 100
        assert self.tearing_history_window % 50 == 0
        self.tearing_start_lookback = -1 - self.tearing_history_window // 50
        headers = [dat[1] for dat in self.tearing_headers] + ['efsbetan']
        if not nn_tearing:
            self.tearing_model = load_cb_from_files(
                    str(self.tearing_path / 'model.cbm'),
                    str(self.tearing_path / 'dranges.pkl'),
                    str(self.tearing_path / 'headers.pkl'),
                    headers)
        else:
            self.tearing_model = NNTearingModel(tearing_path)
        self.tearing_input = None

    def reset(self):
        state = super().reset()
        self.tearabilities = []
        self.current_tearing_prob = None
        self.tearing_input = None
        super().compute_reward(self._state)
        return state

    def make_tearing_input(self, state, beta_n, prev_beta_n):
        tearing_input = []
        theta = ((np.arange(self.tearing_history_window) + 1) / self.tearing_history_window)[np.newaxis, ...]
        for state_name, tear_name in self.tearing_headers:
            start_point = state[state_name][:, self.tearing_start_lookback:self.tearing_start_lookback + 1]
            end_point = state[state_name][:, -1:]
            interpolated_data = start_point * (1 - theta) + end_point * theta
            tearing_input += [interpolated_data]
        if prev_beta_n.ndim == 1:
            prev_beta_n = prev_beta_n[:, np.newaxis]
            beta_n = beta_n[:, np.newaxis]
        beta_data = prev_beta_n * (1 - theta) + beta_n * theta
        tearing_input += [beta_data]
        tearing_input = np.transpose(np.stack(tearing_input), (1, 2, 0))
        return tearing_input

    def compute_reward(self, state):
        denorm_state = denormalize(state, self.normalization_dict, verbose=False)
        old_beta_n = self.current_beta_n
        self.current_beta_n = self._compute_beta_n(denorm_state)
        if old_beta_n.shape != self.current_beta_n.shape:
            old_beta_n = np.tile(old_beta_n[0], self.current_beta_n.shape)
        beta_n_reward = -(self.current_beta_n - self.target_beta_n) ** 2
        self.tearing_input = self.make_tearing_input(denorm_state, self.current_beta_n, old_beta_n)
        self.current_tearing_prob = self.tearing_model.multi_predict(self.tearing_input)
        self.tearabilities.append(self.current_tearing_prob)
        exp_term = np.exp(self.rew_coefs[1] * (self.current_tearing_prob - 0.5))
        dis_loss = self.rew_coefs[0] * (exp_term / (1 + exp_term))
        return beta_n_reward - dis_loss

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        info['tearing_prob'] = self.current_tearing_prob
        new_info = {k: v for k, v in info.items() if v is not None}
        # info['tearing_input'] = self.tearing_input
        return next_state, reward, done, new_info


class ProfileTargetEnv(ProfileEnv):
    def __init__(self, scenario_path, gpu_num=None, **kwargs):
        super().__init__(scenario_path, gpu_num)
        target_profile_name = 'temp'
        # might have to divide these by 1000 if kev units here
        core_value = 3200
        pedestal_value = 800
        edge_value = 0
        pedestal_cutoff = 0.8
        num_points = self.profile_length
        core_values = np.linspace(core_value, pedestal_value, int(num_points * pedestal_cutoff))
        edge_values = np.linspace(pedestal_value, edge_value, int(num_points * (1 - pedestal_cutoff) + 2))[1:]
        self.target_profile = np.concatenate([core_values, edge_values])

    def compute_reward(self, state):
        db()
        denorm_state = denormalize(state, self.normalization_dict, verbose=False)
        profile = self.get_value_from_denorm_state(denorm_state)
        return -(profile - self.target_profile).square().sum()




class ScalarEnv(ProfileEnv):
    def __init__(self, scenario_path, gpu_num=None):
        super().__init__(scenario_path, gpu_num)
        obs_bot = []
        obs_top = []
        for sig in self.actuator_inputs + self.scalar_inputs:
            obs_bot += [self.bounds[sig][0]] * (self.scenario['lookbacks'][sig] + 1)
            obs_top += [self.bounds[sig][1]] * (self.scenario['lookbacks'][sig] + 1)
        obs_bot = np.array(obs_bot)
        obs_top = np.array(obs_top)

        self.observation_space = spaces.Box(low=obs_bot, high=obs_top)


    @property
    def obs(self):
        state = [self._state['input_past_' + sig].flatten() for sig in self.actuator_inputs] + \
                [self._state['input_' + sig].flatten() for sig in self.scalar_inputs]
        # don't use future actuators because they are the action
        # [self._state['input_future_' + sig] for sig in self.actuator_inputs] + \
        return np.concatenate(state)


class NonPhysicalScalarEnv(ScalarEnv):
    def __init__(self, scenario_path, gpu_num=None, **kwargs):
        super().__init__(scenario_path, gpu_num)

    def _compute_beta_n(self, state):
        betan = state['input_betan_EFIT01'][0, 0]
        return betan

    def step(self, action):
        obs, rew, done, wrong_info = super().step(action)
        info = {'beta_n': wrong_info['beta_n']}
        return obs, rew, done, info



class NonPhysicalProfileEnv(ProfileEnv):
    def __init__(self, scenario_path, gpu_num=None):
        super().__init__(scenario_path, gpu_num)

    def _compute_beta_n(self, state):
        betan = state['input_betan_EFIT01'][0, 0]
        return betan


class NonPhysicalTearingProfileEnv(TearingProfileEnv):
    def __init__(self, scenario_path, tearing_path, rew_coefs, gpu_num=None):
        super().__init__(scenario_path, tearing_path, rew_coefs, gpu_num)

    def _compute_beta_n(self, state):
        betan = state['input_betan_EFIT01'][0, 0]
        return betan


def test_env():
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    env.reset()
    rewards = []
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break


def test_rollout():
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    state = env.reset()
    n_actions = 50
    n_steps = 10
    actions = []
    for _ in range(n_actions):
        traj_actions = []
        for _ in range(n_steps):
            traj_actions.append(env.action_space.sample())
        actions.append(traj_actions)
    actions = np.array(actions)
    states = env.unroll(state, actions)
    return states


def test_tearing_env():
    rew_coefs = (1, 1)
    env = TearingProfileEnv(scenario_path=SCENARIO_PATH, tearing_path=TEARING_PATH, rew_coefs=rew_coefs)
    env.reset()
    rewards = []
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break


def test_tearing_rollout():
    rew_coefs = (1, 1)
    env = TearingProfileEnv(scenario_path=SCENARIO_PATH, tearing_path=TEARING_PATH, rew_coefs=rew_coefs)
    state = env.reset()
    n_actions = 50
    n_steps = 10
    actions = []
    for _ in range(n_actions):
        traj_actions = []
        for _ in range(n_steps):
            traj_actions.append(env.action_space.sample())
        actions.append(traj_actions)
    actions = np.array(actions)
    states = env.unroll(state, actions)
    return states

def test_nn_tearing_rollout():
    rew_coefs = (1, 1)
    env = TearingProfileEnv(scenario_path=SCENARIO_PATH,
            tearing_path=NN_TEARING_PATH, rew_coefs=rew_coefs, nn_tearing=True)
    state = env.reset()
    n_actions = 50
    n_steps = 10
    actions = []
    for _ in range(n_actions):
        traj_actions = []
        for _ in range(n_steps):
            traj_actions.append(env.action_space.sample())
        actions.append(traj_actions)
    actions = np.array(actions)
    states = env.unroll(state, actions)
    return states

def compute_tearing_stats():
    rew_coefs = (1, 1)
    env = TearingProfileEnv(scenario_path=SCENARIO_PATH, tearing_path=TEARING_PATH, rew_coefs=rew_coefs)
    tearing_inputs = []
    n_eps = 10
    for nep in trange(n_eps):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            tearing_inputs.append(info['tearing_prob'])
    tearing_inputs = np.concatenate(tearing_inputs, axis=0).reshape(-1, 10)
    max_tearing = tearing_inputs.max(axis=0)
    min_tearing = tearing_inputs.min(axis=0)
    mean_tearing = tearing_inputs.mean(axis=0)
    std_tearing = tearing_inputs.std(axis=0)
    for prof, tear in env.tearing_headers:
        print(tear)
    tearing_data = {}
    for i, proftear in enumerate(env.tearing_headers):
        tear = proftear[1]
        tearing_data[tear] = tearing_inputs[:, i]
    with open('tearing_inputs.pk', 'wb') as f:
        pickle.dump(tearing_data, f)


if __name__ == '__main__':
    test_env()
    test_rollout()
    print(f"completed non-tearing stuff")
    test_tearing_env()
    test_tearing_rollout()
    test_nn_tearing_rollout()
    compute_tearing_stats()
