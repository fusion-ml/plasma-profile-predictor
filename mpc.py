from abc import ABC, abstractmethod

class MPC(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def plan(self, state):
        '''
        state is a 1d ndarray, expect an action out as a 1d ndarray
        '''
        pass

    def __call__(self, state):
        return self.plan(state)

class RS(MPC):
    def __init__(self, env, shots):
        super().__init__(env)
        self.shots = shots

    def plan(self, state):
        pass


class CEM(MPC):
    def __init__(self, env, popsize, n_elites, n_iters):
        pass

    def plan(self, state):
        pass

def test_rs():
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    state = env.reset()
    rs = RS(env, 1000)
    total_reward = 0
    done = False
    while not done:
        action = rs(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f"Rollout finished with total reward {total_reward}")


if __name__ == '__main__':
    test_rs()
