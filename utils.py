import json
from pathlib import Path
import pickle
import numpy as np
from ipdb import set_trace as db


DATA_DIR = Path('data/')
VAL_PATH = Path('/zfsauton/project/public/virajm/plasma_models/val.pkl')


def make_output_dir(name, overwrite, args):
    dir_path = get_output_dir(name)
    if dir_path.exists():
        if overwrite:
            for fn in dir_path.iterdir():
                fn.unlink()
        else:
            raise ValueError(f"{dir_path} already exists! Use a different name")
    else:
        dir_path.mkdir()
    args_name = dir_path / 'args.json'
    args = vars(args)
    with args_name.open('w') as f:
        json.dump(args, f)
    return dir_path


def get_output_dir(name):
    dir_path = DATA_DIR / name
    return dir_path

def get_historical_slice(shotnum, time, valdata):
    # an artifact of the fact that there's a 300ms lookback and 200ms look forward, so the current time is actually
    # at index 6
    present_time_idx = 6
    shot_mask = valdata['shotnum'][:, present_time_idx] == shotnum
    time_mask = valdata['time'][:, present_time_idx] == time
    shot_time_mask = np.logical_and(shot_mask, time_mask)
    if shot_time_mask.any() is False:
        raise ValueError(f"Time {time} in shot {shotnum} not found!")
    if shot_time_mask.sum() > 1:
        raise ValueError(f"There are multiple elements with shot {shotnum} and time {time}")
    idx = np.nonzero(shot_time_mask)
    # data = {key: valdata[key][idx, present_time_idx] for key in valdata}
    data = {}
    for key in valdata:
        if valdata[key].ndim == 3:
            # profiles only go into the future so we take the first one which is the present
            data[key] = valdata[key][idx, 0, :]
        else:
            data[key] = valdata[key][idx, present_time_idx]
            print(data[key].shape)
    return data

if __name__ == '__main__':
    with VAL_PATH.open('rb') as f:
        valdata = pickle.load(f)
    data = get_historical_slice(173228, 1000, valdata)
