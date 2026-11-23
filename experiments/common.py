"""Shared helpers for the experiment scripts: paths, saving, parallel runs.

Every script is run from the root of the repository, e.g.
    .venv\\Scripts\\python experiments\\exp_dp.py
and takes its options as key=value arguments.
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
sys.path.insert(0, str(ROOT / 'src'))


def args(**defaults):
    """Parse key=value arguments, with the types of the defaults."""
    out = dict(defaults)
    for a in sys.argv[1:]:
        k, v = a.split('=', 1)
        if k not in defaults:
            raise SystemExit(f'unknown option {k}; options: {", ".join(defaults)}')
        d = defaults[k]
        if isinstance(d, (list, tuple)):
            typ = type(d[0]) if d else str
            out[k] = [typ(x) for x in v.split(',')]
        elif isinstance(d, bool):
            out[k] = v.lower() in ('1', 'true', 'yes')
        else:
            out[k] = type(d)(v)
    return out


def out_dir(*parts):
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run_grid(fn, configs, processes=None):
    """fn(config) for every config, in parallel; results in order."""
    processes = (processes or int(os.environ.get('N_PROCS', 0))
                 or max(1, (os.cpu_count() or 2) - 2))
    processes = min(processes, len(configs))
    t0 = time.perf_counter()
    with Pool(processes) as pool:
        results = pool.map(fn, configs, chunksize=1)
    print(f'{len(configs)} runs in {time.perf_counter() - t0:.0f} s on {processes} processes')
    return results
