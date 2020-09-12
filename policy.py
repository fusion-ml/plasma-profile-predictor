from abc import ABC, abstractmethod
import numpy as np
from profile_env import ProfileEnv, SCENARIO_PATH


class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state):
        pass

    def reset(self):
        pass


class PID(Policy):
    def __init__(self, P=0.2, I=0.0, D=0.0, tau=0.2, env=None):
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
        betan = self.env.compute_betan(obs)
        action = self.update(betan)
        tinj = 0.5
        target_density = 1.
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
