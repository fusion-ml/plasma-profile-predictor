from abc import ABC, abstractmethod
import numpy as np
import torch
from profile_env import ProfileEnv, SCENARIO_PATH

from helpers.normalization import denormalize
from utils import get_historical_slice

class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state):
        pass

    def reset(self):
        pass


class PINJRLPolicy(Policy):

    def __init__(
            self,
            model_path,
            env,
            cuda_device='',
    ):
        self.device = 'cpu' if cuda_device == '' else 'cuda:' + cuda_device
        self.nn = torch.load(model_path,
                             map_location=self.device)['evaluation/policy']
        self.env = env
        self.betan_param_dict = {
            'betan_EFIT01': env.normalization_dict['betan_EFIT01'],
         }
        self._start_time = None

    def __call__(self, state):
        # Look up the other action variables applied other than pinj.
        other_actions = self._get_other_actions()
        if other_actions is None:
            return None
        # Get the features needed and pass through the neural nent.
        net_in = torch.Tensor(self._get_net_features()).to(self.device)
        with torch.no_grad():
            pinj = float(self.nn.get_action(net_in)[0])
        # Normalize the pinj to the environment.
        # TODO: Should we also be considering the lower bound here???
        pbound = self.env.bounds['pinj']
        pinj = (pinj + 1) / 2
        pinj = (pinj + pbound[0]) / (pbound[1] - pbound[0])
        return np.array([
            other_actions['target_density'],
            other_actions['tinj'],
            pinj,
            other_actions['target_current'],
        ])

    def reset(self):
        """Reset. Assume that the environment was reset before this."""
        # Set absolute start time of the environment.k
        self._start_time = self.env.absolute_time

    def _get_other_actions(self):
        # Viraj's PID returns hardcoded values. Let's try that for now.
        return {
            'tinj': 0.5,
            'target_density': 1.0,
            'target_current': 0.5,
        }
        # Or could use historical data.
        time = self.env.absolute_time
        shotnum = self.env.val_generator.cur_shotnum[0, 0]
        try:
            tslice = get_historical_slice(shotnum, time,
                                          self.env.val_generator.data)
        except ValueError:
            return None
        return {k: float(tslice[k])
                for k in ['curr_target', 'tinj', 'target_density']}

    def _get_net_features(self):
        # Get the beta_n values.
        time = self.env.absolute_time
        shotnum = self.env.val_generator.cur_shotnum[0, 0]
        betas = self.env.betans[-3:]
        insertions = 0
        to_append = []
        while len(betas) + insertions < 3:
            logged_beta = get_historical_slice(
                shotnum,
                time + (insertions - 2) * 200,
                self.env.val_generator.data,
            )['betan_EFIT01'].flatten()
            logged_beta = denormalize(
                    {'input_betan_EFIT01': logged_beta},
                    self.betan_param_dict,
                    verbose=False,
            )['input_betan_EFIT01']
            if len(logged_beta) == 1:
                to_append.append(float(logged_beta))
            insertions += 1
        betas = to_append + betas
        betas = (np.array(betas) - 1.32) / 0.94
        beta_mean = np.mean(betas)
        beta_slope = betas[-1] - betas[0]
        # Get Tearability.
        if len(self.env.tearabilities) == 0:
            tearability = 0.1
        else:
            tearability = np.mean(self.env.tearabilities[-3:])
        return [float(beta_mean), float(beta_slope), float(tearability)]


class PID(Policy):
    def __init__(self, env=None, P=0.2, I=0.0, D=0.0, tau=0.2):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.tau = tau
        self.env = env

        self.sample_time = 0.0
        self.current_time = 0.
        self.reset()

    def reset(self):
        self.PTerm = 0.
        self.ITerm = 0.
        self.DTerm = 0.
        self.SetPoint = 1.5
        self.last_error = 0.
        self.int_error = 0.
        self.windup_guard = 20.

        self.output = 0.

    def update(self, feedback_value):
        error = self.SetPoint - feedback_value

        # self.current_time = current_time if current_time is not None else tim
        delta_time = self.tau
        delta_error = error - self.last_error
        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)
            return self.output

    def __call__(self, obs):
        betan = self.env.compute_beta_n(obs)
        action = self.update(betan)
        tinj = 0.5
        target_density = 1.0
        target_current = 0.5
        pinj = action
        action = np.array([target_density, tinj, pinj, target_current])
        return action


def test_pid():
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    state = env.reset()
    pid = PID(env=env)
    total_reward = 0
    done = False
    while not done:
        action = pid(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Rollout finished with total reward {total_reward}")


if __name__ == '__main__':
    test_pid()
