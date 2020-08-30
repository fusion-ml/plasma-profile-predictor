import os
from gym import Env, spaces
import keras
import pickle
import numpy as np

from helpers.data_generator import DataGenerator
from helpers.normalization import denormalize
from tearing.disrupt_predictor import load_cb_from_files
from ipdb import set_trace as db


SCENARIO_PATH = "/zfsauton2/home/virajm/src/plasma-profile-predictor/outputs/beta_n_signals/model-conv2d_profiles-dens-temp-q_EFIT01-rotation-press_EFIT01_act-target_density-pinj-tinj-curr_target_30Jul20-16-13_params.pkl"  # NOQA


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
        self.target_beta_n = 1.5
        self.bounds = {
                'a_EFIT01': (-1.8, 1.95),
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
        self._model = keras.models.load_model(model_path, compile=False)
        self._state = None
        self._state = None
        self.t = None
        self.timestep = 50
        self.t_max = 2200
        self.i = 0
        self.earliest_start_time = 500
        self.latest_start_time = 1000
        self.mu_0 = 1.256637E-6
        self.current_beta_n = None

    def reset(self):
        while True:
            example = self.val_generator[self.i][0]
            time = self.val_generator.cur_times[0, self.time_lookback]
            if time > self.earliest_start_time and time < self.latest_start_time:
                # TODO: arrange the data in state, return it
                self._state = example
                self.t = 0
                return self.obs
            self.i += 1
            if self.i == len(self.val_generator):
                print("Went through whole generator, restarting.")
                self.i = 0
        self.current_beta_n = None

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
        done = self.t > self.t_max
        return self.obs, reward, done, {'beta_n': self.current_beta_n}

    def compute_beta_n(self, state):
        pressure_profile = state['input_press_EFIT01']  # Pa
        mean_total_field_strength = np.abs(state['input_bt'][..., -1])
        # Here we're making the assumption that B ~= B_t as
        # most of the magnetic field is composed of the
        # toroidal component. Also denoted in Tesla.
        mean_total_field_strength = np.maximum(mean_total_field_strength, 0)
        mean_plasma_pressure = np.mean(pressure_profile, axis=-1)  # TODO: take the geometry of the torus into account
        mean_plasma_pressure = np.maximum(mean_plasma_pressure, 0)
        beta = mean_plasma_pressure * 2 * self.mu_0 / mean_total_field_strength ** 2
        minor_radius = state['input_a_EFIT01'][..., -1]  # meters
        minor_radius = np.maximum(minor_radius, 0)
        current = state['input_curr'][..., -1] / 1e6  # convert to MA from amps
        current = np.maximum(current, 1e-8)  # use 1e-8 for numerical stability
        beta_n = beta * minor_radius * mean_total_field_strength / current
        return beta_n

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
        self.current_beta_n = self.compute_beta_n(denorm_state)
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
    def __init__(self, scenario_path, tearing_path, rew_coefs):
        super().__init__(scenario_path)
        self.tearing_path = tearing_path
        self.current_tearing_prob = None
        self.rew_coefs = rew_coefs
        self.tearing_model = load_cb_from_files(
                self.tearing_path / 'model.cbm',
                self.tearing_path / 'dranges.pkl',
                self.tearing_path / 'headers.pkl',
                data_in_columns)  # dunno what this is supposed to do

    def reset(self):
        super().reset()
        self.current_tearing_prob = None

    def compute_reward(self, state):
        denorm_state = denormalize(state, self.normalization_dict, verbose=False)
        beta_n = self.compute_beta_n(denorm_state)

        beta_n_reward = -(beta_n - self.target_beta_n) ** 2
        self.current_tearing_prob = self.tearing_model(something)
        exp_term = np.exp(self.rew_coefs[1] * (self.current_tearing_prob - 0.5))
        dis_loss = self.rew_coefs[0] * (exp_term / (1 + exp_term))
        return beta_n_reward - dis_loss

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        info['tearing_prob'] = self.current_tearing_prob
        return next_state, reward, done, info


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


if __name__ == '__main__':
    test_env()
    test_rollout()
