from abc import ABC, abstractmethod
from profile_env import ProfileEnv, SCENARIO_PATH
import numpy as np
from scipy import stats
from ipdb import set_trace as db


class MPC(ABC):
    def __init__(self, env, horizon, discount_rate=1):
        self.env = env
        self.horizon = horizon
        self.discount_rate = discount_rate
        self.discount_array = np.power(self.discount_rate, range(horizon))

    @abstractmethod
    def plan(self, state):
        '''
        state is a 1d ndarray, expect an action out as a 1d ndarray
        '''
        pass

    def __call__(self, state):
        return self.plan(state)

    def reset(self):
        pass


class RS(MPC):
    def __init__(self, env, horizon, shots):
        super().__init__(env, horizon)
        self.shots = shots

    def plan(self, state):
        action_sequences = []
        for seqnum in range(self.shots):
            sequence = []
            for step in range(self.horizon):
                sequence.append(self.env.action_space.sample())
            action_sequences.append(sequence)
        action_sequences = np.array(action_sequences)
        obs_sequence, rew_sequence = self.env.unroll(state, action_sequences)
        discounted_rewards = rew_sequence * self.discount_array
        returns = np.sum(discounted_rewards, axis=1)
        best_action_sequence_idx = np.argmax(returns)
        action = action_sequences[best_action_sequence_idx, 0, :]
        return action


class CEM(MPC):
    def __init__(self, env, horizon, popsize, n_elites, n_iters, alpha=0.25, epsilon=1e-3):
        super().__init__(env, horizon)
        self.popsize = popsize
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.alpha = alpha
        self.epsilon = epsilon
        self.ac_ub = self.env.action_space.high
        self.ac_lb = self.env.action_space.low
        self.action_dim = len(self.ac_ub)
        self.sol_dim = self.action_dim * self.horizon
        self.init_mean = None
        self.init_var = None
        self.best_action_sequence = None
        self.best_action_reward = -np.inf
        self.reset()

    def reset(self):
        self.init_mean = np.tile((self.ac_ub + self.ac_lb) / 2, (self.horizon, 1))
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 2, [self.horizon, 1])

    def plan(self, state):
        mean, var, t = self.init_mean, self.init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        self.best_action_sequence = None
        self.best_action_cost = np.inf

        while (t < self.n_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.ac_lb, self.ac_ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.horizon, self.action_dim]) * np.sqrt(constrained_var) + mean


            obs_sequence, rew_sequence = self.env.unroll(state, samples)
            discounted_rewards = rew_sequence * self.discount_array
            costs = -np.sum(discounted_rewards, axis=1)

            elites = samples[np.argsort(costs)[:self.n_elites], ...]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            if costs.min() < self.best_action_cost:
                self.best_action_cost = costs.min()
                self.best_action_sequence = elites[0, ...]

            t += 1
        self.init_mean = mean
        return self.best_action_sequence[0, ...]


def test_rs():
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    state = env.reset()
    rs = RS(env, 10, 1000)
    total_reward = 0
    done = False
    while not done:
        action = rs(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Rollout finished with total reward {total_reward}")

def test_cem():
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    state = env.reset()
    cem = CEM(env, 10, 100, 10, 10)
    total_reward = 0
    done = False
    while not done:
        action = cem(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Rollout finished with total reward {total_reward}")


if __name__ == '__main__':
    test_cem()
