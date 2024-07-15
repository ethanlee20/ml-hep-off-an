
from pathlib import Path
import pandas as pd


def load(dc9, trial):
    filename = f"dc9_{dc9}_{trial}_re.pkl"
    filepath = Path("../datafiles").joinpath(filename)
    result = pd.read_pickle(filepath)
    return result


def load_all_trials(dc9):
    filepaths=list(Path("../datafiles").glob(f"dc9_{dc9}_*_re.pkl"))
    dfs = [pd.read_pickle(path) for path in filepaths]
    result = pd.concat(dfs)
    return result


def file_info(filepath):
    """Accecpts name or path."""
    filepath = Path(filepath)
    split_name = filepath.name.split('_')
    dc9, trial = float(split_name[1]), int(split_name[2])
    result = {'dc9': dc9, 'trial': trial}
    return result


def list_dc9():
    filenames = list(Path("../datafiles").glob("*.pkl"))
    dc9s = [file_info(name)["dc9"] for name in filenames]
    result = sorted(set(dc9s))
    return result 