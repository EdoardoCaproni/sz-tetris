"""Write the data files of the page (viewer/) from the results.

  viewer/data_6x<H>.js          table of the boards (sorted keys), W* (float16),
                                optimal placements, linear weights
  viewer/data_6x<H>_tabular.js  the placements chosen by each tabular method,
                                and the afterstate values of TD(0)
  viewer/data_10x20.js          linear weights
  viewer/results.js             every number and curve the page draws

For each method and board the page plays the run whose final value is the
median over the seeds (a typical run, not the luckiest). Before writing,
every exported policy is evaluated again from what is written and must
give the value stored in its final.json.

Run: .venv\\Scripts\\python experiments\\export_viewer.py
     (after exp_dp.py, exp_tabular.py, exp_linear.py, exp_heuristic.py and
     exp_tabular_10x20.py)
"""
import base64
import json
from collections import defaultdict

import numpy as np

from common import RESULTS, ROOT, load_csv, load_json

import dp
import linear
import tabular

VIEWER = ROOT / 'viewer'
HEIGHTS = (4, 5, 6)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


def f16(a):
    return b64(np.asarray(a, dtype='<f2'))


def letters(pol):
    return (pol.astype(np.uint8).ravel() + 97).tobytes().decode('ascii')


def write_js(name, var, key, obj):
    text = (f'window.{var} = window.{var} || {{}};\n'
            f'window.{var}[{json.dumps(key)}] = {json.dumps(obj, separators=(",", ":"))};\n')
    (VIEWER / name).write_text(text, encoding='utf-8')
    print(f'{name}: {len(text) / 1e6:.2f} MB')


def check(name, got, want, tol=1e-6):
    if abs(got - want) > tol:
        raise SystemExit(f'{name}: exported policy gives {got}, final.json says {want}')


def band(curves, key):
    """Mean, min and max over seeds of the curves (aligned by checkpoint)."""
    n = min(len(c) for c in curves)
    a = np.array([[float(r[key]) for r in c[:n]] for c in curves])
    return dict(mean=a.mean(0).round(5).tolist(), lo=a.min(0).round(5).tolist(),
                hi=a.max(0).round(5).tolist())


def median_run(runs):
    """The run with the median final value (runs: list of (value, payload))."""
    runs = sorted(runs, key=lambda r: r[0])
    return runs[(len(runs) - 1) // 2]


def main():
    VIEWER.mkdir(exist_ok=True)
    res = dict(dp=load_json(RESULTS / 'dp' / 'summary.json'))

    # ------------------------------------------------------------ tabular
    grid = load_csv(RESULTS / 'tabular' / 'grid' / 'summary.csv')
    g = defaultdict(list)
    for r in grid:
        g[(r['method'], float(r['alpha']), float(r['eps']), float(r['q0']))].append(float(r['value']))
    res['tabular_grid'] = [dict(method=k[0], alpha=k[1], eps=k[2], optimistic=k[3] > 0,
                                mean=float(np.mean(v)), lo=min(v), hi=max(v))
                           for k, v in sorted(g.items())]
    tab_final = defaultdict(dict)
    tab_curves = defaultdict(dict)
    shown = defaultdict(dict)
    for H in HEIGHTS:
        t = dp.solve(6, H)
        nxt, rew = t['nxt'], t['rew']
        data = dict(W=6, H=H, n=int(t['keys'].size),
                    keysLo=b64((t['keys'] & 0xFFFFFFFF).astype('<u4')),
                    keysHi=b64((t['keys'] >> 32).astype('u1')),
                    wstar=f16(t['Wstar']), opt=letters(t['pol']), linear={})
        check(f'optimal 6x{H}', dp.evaluate_policy(nxt, rew, t['pol'])[0], t['Wstar'][0])
        extra = dict(tabular={}, tdW=None)
        for m in tabular.METHODS:
            runs = []
            curves = []
            for d in sorted((RESULTS / 'tabular' / 'final').glob(f'{m}_H{H}_s*')):
                fin = load_json(d / 'final.json')
                runs.append((fin['value'], d, fin))
                curves.append(load_csv(d / 'train.csv'))
            if not runs:
                continue
            vals = [r[0] for r in runs]
            fins = [r[2] for r in runs]
            tab_final[H][m] = dict(
                value=float(np.mean(vals)), lo=min(vals), hi=max(vals),
                ratio=float(np.mean(vals) / t['Wstar'][0]),
                agreement=float(np.mean([f['agreement'] for f in fins])),
                pieces=float(np.mean([f['pieces'] for f in fins])),
                seconds=float(np.mean([f['train_seconds'] for f in fins])),
                alpha=fins[0]['alpha'], eps=fins[0]['eps'], q0=fins[0]['q0'],
                episodes=fins[0]['episodes'], seeds=len(runs))
            cur = dict(episodes=[int(r['episodes']) for r in curves[0]], value=band(curves, 'value'))
            if m == 'td_afterstate':
                cur['w_rmse'] = band(curves, 'w_rmse')
            tab_curves[H][m] = cur
            value, d, fin = median_run(runs)
            learned = np.load(d / 'learned.npz')
            pol = learned['pol']
            check(f'{m} 6x{H}', dp.evaluate_policy(nxt, rew, pol, tol=1e-8)[0], value, 1e-5)
            extra['tabular'][m] = letters(pol)
            if m == 'td_afterstate':
                extra['tdW'] = f16(learned['W'])
            shown[f'6x{H}'][m] = dict(value=value, seed=fin['seed'])
        # linear weights for this board, if trained on it
        for m in ('td_linear', 'reinforce'):
            sel = select_linear(m, f'6x{H}')
            if sel is None:
                continue
            value, d, fin = sel
            learned = np.load(d / 'learned.npz')
            Phi = linear.table_features(t, 6, H)
            got = linear.evaluate(m, 6, H, learned['w'], learned['theta'], t, Phi)['value']
            check(f'{m} 6x{H}', got, value, 1e-5)
            data['linear'][m] = (learned['w'] if m == 'td_linear' else learned['theta']).tolist()
            shown[f'6x{H}'][m] = dict(value=value, seed=fin['seed'], alpha=fin['alpha'])
        write_js(f'data_6x{H}.js', 'SZ_DATA', f'6x{H}', data)
        write_js(f'data_6x{H}_tabular.js', 'SZ_TABULAR', f'6x{H}', extra)
    res['tabular_final'] = tab_final
    res['tabular_curves'] = tab_curves

    # ------------------------------------------------------------- linear
    lin = defaultdict(dict)
    data10 = dict(W=10, H=20, linear={})
    for b in ('6x5', '6x6', '10x20'):
        for m in ('td_linear', 'reinforce'):
            runs = collect_linear(m, b)
            if not runs:
                continue
            by_alpha = defaultdict(list)
            for value, d, fin in runs:
                by_alpha[fin['alpha']].append((value, d, fin))
            per_alpha = {a: float(np.mean([r[0] for r in rs])) for a, rs in by_alpha.items()}
            best_a = max(per_alpha, key=per_alpha.get)
            rs = by_alpha[best_a]
            curves = [load_csv(r[1] / 'train.csv') for r in rs]
            vals = [r[0] for r in rs]
            entry = dict(alpha=best_a, per_alpha=per_alpha, value=float(np.mean(vals)),
                         lo=min(vals), hi=max(vals), seeds=len(rs),
                         seconds=float(np.mean([r[2]['train_seconds'] for r in rs])),
                         steps=[int(float(x['steps'])) for x in curves[0]],
                         curve=band(curves, 'value'))
            if b == '10x20':
                entry['ci95'] = float(np.mean([r[2].get('ci95', 0) for r in rs]))
                entry['pieces'] = float(np.mean([r[2].get('pieces', 0) for r in rs]))
                value, d, fin = median_run(rs)
                learned = np.load(d / 'learned.npz')
                data10['linear'][m] = (learned['w'] if m == 'td_linear' else learned['theta']).tolist()
                shown['10x20'][m] = dict(value=value, seed=fin['seed'], alpha=best_a,
                                         ci95=fin.get('ci95'))
            else:
                H = int(b.split('x')[1])
                entry['ratio'] = entry['value'] / load_json(RESULTS / 'dp' / 'summary.json')[H - 3]['optimal']
            lin[b][m] = entry
    base_path = RESULTS / 'linear' / 'baselines.json'
    if base_path.exists():
        res['baselines_10x20'] = load_json(base_path)
    if (RESULTS / 'linear' / 'plateau.json').exists():
        res['plateau_10x20'] = load_json(RESULTS / 'linear' / 'plateau.json')

    # ---------------------------------------------------------- heuristic
    hp = RESULTS / 'heuristic' / 'summary.json'
    if hp.exists():
        heur = load_json(hp)
        heur['10x20']['value'] = heur['10x20']['mean']
        for H in HEIGHTS:
            t = dp.solve(6, H)
            pol = linear.linear_policy_on_table(t, linear.table_features(t, 6, H),
                                                np.array(heur[f'6x{H}']['weights']))
            check(f'heuristic 6x{H}', dp.evaluate_policy(t['nxt'], t['rew'], pol)[0],
                  heur[f'6x{H}']['value'])
        res['heuristic'] = {b: v for b, v in heur.items() if b != '6x3'}

    # ------------------------------------------------- tabular on 10x20
    t20 = {}
    for m in tabular.METHODS:
        dirs = sorted((RESULTS / 'tabular_10x20').glob(f'{m}_s*'))
        dirs = [d for d in dirs if (d / 'final.json').exists()]
        if not dirs:
            continue
        fins = [load_json(d / 'final.json') for d in dirs]
        curves = [load_csv(d / 'train.csv') for d in dirs]
        vals = [f['value'] for f in fins]
        t20[m] = dict(value=float(np.mean(vals)), lo=min(vals), hi=max(vals),
                      ci95=float(np.mean([f['ci95'] for f in fins])),
                      boards=float(np.mean([f['boards'] for f in fins])),
                      coverage=float(np.mean([f['coverage'] for f in fins])),
                      pieces=float(np.mean([f['pieces'] for f in fins])),
                      steps=fins[0]['steps'], seeds=len(fins), alpha=fins[0]['alpha'],
                      eps=fins[0]['eps'], seconds=float(np.mean([f['train_seconds'] for f in fins])),
                      curve=dict(steps=[int(float(r['steps'])) for r in curves[0]],
                                 value=band(curves, 'value'), coverage=band(curves, 'coverage'),
                                 boards=band(curves, 'boards')))
    if t20:
        res['tabular_10x20'] = t20
    res['linear'] = lin
    res['shown'] = shown
    write_js('data_10x20.js', 'SZ_DATA', '10x20', data10)
    (VIEWER / 'results.js').write_text(
        'window.SZ_RESULTS = ' + json.dumps(res, separators=(',', ':')) + ';\n', encoding='utf-8')
    print('results.js written')


def collect_linear(method, board):
    runs = []
    for d in sorted((RESULTS / 'linear').glob(f'{method}_{board}_a*_s*')):
        if (d / 'final.json').exists():
            fin = load_json(d / 'final.json')
            if np.isfinite(fin.get('value', np.nan)):
                runs.append((fin['value'], d, fin))
    return runs


def select_linear(method, board):
    runs = collect_linear(method, board)
    if not runs:
        return None
    by_alpha = defaultdict(list)
    for r in runs:
        by_alpha[r[2]['alpha']].append(r)
    best = max(by_alpha, key=lambda a: np.mean([r[0] for r in by_alpha[a]]))
    return median_run(by_alpha[best])


if __name__ == '__main__':
    main()
