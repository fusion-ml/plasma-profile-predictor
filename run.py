import argparse
from tqdm import trange, tqdm
import pickle
from profile_env import ProfileEnv, TearingProfileEnv, SCENARIO_PATH, TEARING_PATH
from policy import PID, PINJRLPolicy
from mpc import CEM, RS
from utils import make_output_dir


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="The name of the experiment and output directory")
    parser.add_argument("--policy", default="RS", choices=["RS", "CEM", "PID", "RL"])
    parser.add_argument("--num_trials", type=int, default=100, help="The number of rollouts to conduct")
    parser.add_argument("--num_samples", type=int, default=1000, help="The number of samples in an RS run")
    parser.add_argument("--popsize", type=int, default=100, help="The population size for CEM")
    parser.add_argument("--num_elites", type=int, default=10, help="The number of elites in a CEM run")
    parser.add_argument("--num_iters", type=int, default=10, help="The number of iterations of CEM to run")
    parser.add_argument("--discount_rate", type=float, default=1., help="The discount rate for optimization purposes")
    parser.add_argument("--horizon", type=int, default=5, help="The horizon for optimization")
    parser.add_argument("--alpha_cem", type=float, default=0.25, help="The alpha for CEM")
    parser.add_argument("--epsilon_cem", type=float, default=0.01, help="The epsilon for CEM")
    parser.add_argument("--env", default="full", choices=["full", "betan"])
    parser.add_argument("-ow", dest="overwrite", action="store_true")
    parser.add_argument("-P", type=float, default=0.2, help="Proportional gain")
    parser.add_argument("-I", type=float, default=0.0, help="Integral gain")
    parser.add_argument("-D", type=float, default=0.0, help="Derivative gain")
    parser.add_argument('--rl_model_path', help='Path to policy.')
    parser.add_argument('--cuda_device', default='')
    parser.add_argument('--pudb', action='store_true')
    return parser.parse_args()


def run_trial(policy, env):
    state = env.reset()
    policy.reset()
    states = [state]
    actions = []
    rewards = []
    infos = []
    done = False
    while not done:
        action = policy(state)
        if actions is None:
            break
        state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        tqdm.write(f"Action Reward: {reward}")
        rewards.append(reward)
        infos.append(info)
    tqdm.write(f"Total Reward: {sum(rewards)}")
    return states, actions, rewards, infos

def create_env(args):
    if args.env == "betan":
        env = ProfileEnv(scenario_path=SCENARIO_PATH)
    elif args.env == "full":
        rew_coefs = (9, 10)
        env = TearingProfileEnv(scenario_path=SCENARIO_PATH,
                                tearing_path=TEARING_PATH,
                                rew_coefs=rew_coefs)
    return env

def main(args):
    if args.pudb:
        import pudb; pudb.set_trace()
    output_dir = make_output_dir(args.name, args.overwrite, args)
    env = create_env(args)
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
    elif args.policy == "PID":
        policy = PID(env=env,
                     P=args.P,
                     I=args.I,
                     D=args.D,
                     tau=env.tau)
    elif args.policy == 'RL':
        policy = PINJRLPolicy(
            model_path=args.rl_model_path,
            env=env,
            cuda_device=args.cuda_device,
        )
    else:
        raise ValueError('Unknown policy: %s' % args.policy)
    episodes = []
    episode_path = output_dir / 'episodes.pk'
    for i in trange(args.num_trials):
        states, actions, rewards, infos = run_trial(policy, env)
        episodes.append((states, actions, rewards, infos))
        with episode_path.open('wb') as f:
            pickle.dump(episodes, f)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
