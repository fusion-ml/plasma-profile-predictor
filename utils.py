import json
from pathlib import Path

DATA_DIR = Path('data/')


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
