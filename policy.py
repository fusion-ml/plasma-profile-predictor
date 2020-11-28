from abc import ABC, abstractmethod
import numpy as np
import torch
from simple_pid import PID
from profile_env import ProfileEnv, SCENARIO_PATH

from helpers.normalization import denormalize
from utils import get_historical_slice

DEFAULT_OTHER_ACTIONS = {
    'tinj': np.array([0.5]),
    'target_density': np.array([1.0]),
    'curr_target': np.array([0.5]),
}

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
        pinj = pinj * (pbound[1] - pbound[0]) + pbound[0]
        return np.array([
            other_actions['target_density'],
            other_actions['tinj'],
            pinj,
            other_actions['curr_target'],
        ])

    def reset(self):
        """Reset. Assume that the environment was reset before this."""
        # Set absolute start time of the environment.k
        self._start_time = self.env.absolute_time

    def _get_other_actions(self):
        tslice= {
            'tinj': np.array([]),
            'target_density': np.array([]),
            'curr_target': np.array([]),
        }
        time = self.env.absolute_time
        shotnum = self.env.val_generator.cur_shotnum[0, 0]
        try:
            t = time
            while len(tslice['curr_target'].flatten()) == 0 and t > 0:
                tslice = get_historical_slice(shotnum, time,
                                              self.env.val_generator.data)
                t -= self.env.timestep
            if len(tslice['curr_target'].flatten()) == 0:
                return DEFAULT_OTHER_ACTIONS
        except ValueError:
            return None
        return {k: float(tslice[k])
                for k in ['curr_target', 'tinj', 'target_density']}

    def _get_net_features(self):
        # Get the beta_n values.
        time = self.env.absolute_time
        shotnum = self.env.val_generator.cur_shotnum[0, 0]
        betas = [float(b) for b in self.env.betans[-3:]]
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


class PIDPolicy(Policy):
    def __init__(self, env, P=0.2, I=0.0, D=0.0, tau=0.2):
        self.env = env
        self.dt = tau
        self.pid = PID(P, I, D, setpoint=1.5, output_limits=(-1.8, 2.5))

    def reset(self):
        self.pid.reset()

    def __call__(self, obs):
        other_actions = self._get_other_actions()
        if other_actions is None:
            return None
        betan = self.env.compute_beta_n(obs)
        pinj = self.pid(betan, dt=self.dt)
        return np.array([
            other_actions['target_density'],
            other_actions['tinj'],
            pinj,
            other_actions['curr_target'],
        ])

    def _get_other_actions(self):
        tslice= {
            'tinj': np.array([]),
            'target_density': np.array([]),
            'curr_target': np.array([]),
        }
        time = self.env.absolute_time
        shotnum = self.env.val_generator.cur_shotnum[0, 0]
        try:
            t = time
            while len(tslice['curr_target'].flatten()) == 0 and t > 0:
                tslice = get_historical_slice(shotnum, time,
                                              self.env.val_generator.data)
                t -= self.env.timestep
            if len(tslice['curr_target'].flatten()) == 0:
                return DEFAULT_OTHER_ACTIONS
        except ValueError:
            return None
        return {k: float(tslice[k])
                for k in ['curr_target', 'tinj', 'target_density']}


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
