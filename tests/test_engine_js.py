"""The page's engine (viewer/engine.js) against the Python one: same drops,
same keys, same features on random boards. Needs Node; skipped without it."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tetris import placements, n_placements, drop, pack, features, n_features, GAME_OVER

HERE = Path(__file__).resolve().parent


def random_board(rng, W, H):
    rows = np.zeros(H, dtype=np.int64)
    fill = int(rng.integers(0, H))          # a random number of low rows
    for r in range(fill):
        rows[r] = int(rng.integers(0, (1 << W) - 1))   # never full
    return rows


@pytest.mark.skipif(shutil.which('node') is None, reason='node not installed')
def test_engine_js_matches_python(tmp_path):
    rng = np.random.default_rng(5)
    cases = []
    for W, H, n in ((6, 5, 300), (6, 6, 300), (10, 20, 300)):
        cells = placements(W)[0]
        for _ in range(n):
            rows = random_board(rng, W, H)
            f = np.zeros(n_features(W))
            features(rows, W, H, f)
            results = []
            for p in range(2):
                res = []
                for a in range(n_placements(W)):
                    tmp = rows.copy()
                    c = drop(tmp, H, (1 << W) - 1, cells[p, a])
                    res.append(dict(cleared=int(c), rows=[int(x) for x in tmp]))
                results.append(res)
            case = dict(W=W, H=H, rows=[int(x) for x in rows],
                        features=[float(x) for x in f], results=results)
            if W * H <= 63:
                case['key'] = int(pack(rows, W))
            cases.append(case)
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps(cases))
    out = subprocess.run(['node', str(HERE / 'check_engine.js'), str(path)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().split('\n')[-1] == '0', out.stdout[-2000:]
