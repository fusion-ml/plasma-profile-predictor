import argparse
from tqdm import trange
import pickle
from profile_env import ProfileEnv, SCENARIO_PATH
from mpc import CEM, RS
from utils import make_output_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="The name of the experiment and output directory")
    parser.add_argument("--policy", default="RS", choices=["RS", "CEM"])
    parser.add_argument("--num_trials", type=int, default=100, help="The number of rollouts to conduct")
    parser.add_argument("--num_samples", type=int, default=1000, help="The number of samples in an RS run")
    parser.add_argument("--popsize", type=int, default=100, help="The population size for CEM")
    parser.add_argument("--num_elites", type=int, default=10, help="The number of elites in a CEM run")
    parser.add_argument("--num_iters", type=int, default=10, help="The number of iterations of CEM to run")
    parser.add_argument("--discount_rate", type=float, default=1., help="The discount rate for optimization purposes")
    parser.add_argument("--horizon", type=int, default=10, help="The horizon for optimization")
    parser.add_argument("--alpha_cem", type=float, default=0.25, help="The alpha for CEM")
    parser.add_argument("--epsilon_cem", type=float, default=0.01, help="The epsilon for CEM")
    parser.add_argument("-ow", name="overwrite", action="store_true")


def run_trial(policy, env):
    state = env.reset()
    policy.reset()
    states = [state]
    actions = []
    rewards = []
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards


def main(args):
    output_dir = make_output_dir(args.name, args.overwrite, args)
    env = ProfileEnv(scenario_path=SCENARIO_PATH)
    if args.policy == "RS":
        policy = RS(env=env,
                    horizon=args.horizon,
                    shots=args.num_samples)
    elif args.policy == "CEM":
        policy = CEM(env,
                     horizon=args.horizon,
                     popsize=args.popsize,
                     n_elites=args.num_elites,
                     n_iters=args.num_iters,
                     alpha=args.alpha_cem,
                     epsilon=args.epsilon_cem)
    episodes = []
    for i in trange(args.num_trials):
        states, actions, rewards = run_trial(policy, env)
        episodes.append((states, actions, rewards))
    episode_path = output_dir / 'epsiodes.pk'
    with episode_path.open('wb') as f:
        pickle.dump(episodes, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
