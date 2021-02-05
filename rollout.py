from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.envs.wrappers import NormalizedBoxEnv
# from profile_env import ProfileTargetEnv, SCENARIO_PATH
import argparse
import torch
import pickle
import uuid
from helpers.gym_http_client import ClientWrapperEnv
from rlkit.core import logger
from ipdb import set_trace as db

filename = 'paths.pkl'


def simulate_policy(args):
    data = torch.load(args.file, map_location=torch.device('cpu'))
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    remote_base = 'http://127.0.0.1:5000'
    env_id = 'profile-target-env-v0'
    env = NormalizedBoxEnv(ClientWrapperEnv(remote_base, env_id))


    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    paths = []
    for _ in range(10):
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        db()
        logger.dump_tabular()
        paths.append(path)
    with open(filename, 'wb') as f:
        pickle.dump(paths, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=20,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
