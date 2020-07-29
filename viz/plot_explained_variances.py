import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from ipdb import set_trace as db


def main(params_fn, signal_name='explained_variance_score'):
    with open(params_fn, 'rb') as f:
        data = pickle.load(f)
    history = data['history']
    explained_variances = {}
    for key, value in history.items():
        if not (key.endswith(signal_name) and key.startswith('val')):
            continue
        name = key[11:]
        name = name[:-(len(signal_name) + 1)]
        if name.endswith('EFIT01'):
            name = name[:-7]
        explained_variances[name] = value[-1]

    plt.figure(figsize=(12, 6))
    plt.bar(explained_variances.keys(), explained_variances.values())
    plt.title(f"{signal_name} on Validation Set")
    fn = os.path.splitext(params_fn)[0] + f'_{signal_name}.png'
    plt.savefig(fn)


if __name__ == '__main__':
    main(*sys.argv[1:])
