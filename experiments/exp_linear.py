"""Linear function approximation and REINFORCE: 6x5 and 6x6 (exact
evaluation, so the loss due to the features is measured exactly) and
10x20 (Monte Carlo evaluation), plus the baselines on 10x20.

Grid, the same on every board:
  td_linear  alpha in {0.001, 0.003, 0.01}, epsilon = 0.01
  reinforce  alpha_theta in {0.001, 0.003, 0.01, 0.03, 0.1}, alpha_w = 0.01
(the last two REINFORCE values were added after 0.01 came out best; run
them alone with methods=reinforce alphas=0.03,0.1: the summary keeps the
runs already there).
Each setting with `seeds` seeds; budgets in pieces placed (steps).
On 10x20 the final policy of every run is re-evaluated with `final_games`
games (95% confidence interval).

Output: results/linear/<method>_<W>x<H>_a<alpha>_s<seed>/{config.json,train.csv,final.json,learned.npz}
        results/linear/summary.csv, results/linear/baselines.json
Run:    .venv\\Scripts\\python experiments\\exp_linear.py [boards=6x5,6x6,10x20]
"""
import numpy as np

from common import args, out_dir, run_grid, save_csv, save_json, load_csv

import dp
import linear
import simulate

ALPHAS = {'td_linear': (0.001, 0.003, 0.01), 'reinforce': (0.001, 0.003, 0.01, 0.03, 0.1)}
STEPS = {'6x5': 20_000_000, '6x6': 20_000_000, '10x20': 50_000_000}


def run_one(cfg):
    W, H = cfg['W'], cfg['H']
    table = dp.solve(W, H) if W * H <= 36 else None
    curve, final, learned = linear.train(
        cfg['method'], W, H, cfg['steps'], cfg['alpha'], eps=cfg['eps'],
        alpha_w=cfg['alpha_w'], seed_=cfg['seed'], table=table,
        eval_games=cfg['eval_games'])
    if table is None and np.isfinite(learned['w']).all() and np.isfinite(learned['theta']).all():
        if cfg['method'] == 'td_linear':
            r = simulate.monte_carlo(simulate.LINEAR, W, H, cfg['final_games'],
                                     seed=999, weights=learned['w'])
        else:
            r = simulate.monte_carlo(simulate.SOFTMAX, W, H, cfg['final_games'],
                                     seed=999, weights=learned['theta'])
        final.update(value=r['mean'], ci95=r['ci95'], pieces=r['pieces'],
                     capped=r['capped'], games=cfg['final_games'])
    final.update({k: v for k, v in cfg.items() if k != 'save'})
    d = out_dir(*cfg['save'])
    save_json(d / 'config.json', cfg)
    save_csv(d / 'train.csv', curve)
    save_json(d / 'final.json', final)
    np.savez_compressed(d / 'learned.npz', **learned)
    return final


def main():
    opt = args(boards=['6x5', '6x6', '10x20'], seeds=5, final_games=2000,
               methods=['td_linear', 'reinforce'], alphas=[0.0])
    cfgs = []
    for b in opt['boards']:
        W, H = map(int, b.split('x'))
        if W * H <= 36:
            dp.solve(W, H, verbose=True)
        for m in opt['methods']:
            for a in ALPHAS[m]:
                if opt['alphas'] != [0.0] and a not in opt['alphas']:
                    continue
                for s in range(opt['seeds']):
                    cfgs.append(dict(method=m, W=W, H=H, alpha=a, alpha_w=0.01,
                                     eps=0.01 if m == 'td_linear' else 0.0,
                                     steps=STEPS[b], seed=s, eval_games=200,
                                     final_games=opt['final_games'],
                                     save=['linear', f'{m}_{b}_a{a}_s{s}']))
    # the long 10x20 runs first, so they do not finish last; the 6x6 runs
    # hold the table (several hundred MB each), so fewer at a time
    cfgs.sort(key=lambda c: -c['steps'])
    small = [c for c in cfgs if c['W'] * c['H'] != 36]
    large = [c for c in cfgs if c['W'] * c['H'] == 36]
    rows = (run_grid(run_one, small) if small else []) + \
           (run_grid(run_one, large, processes=8) if large else [])
    path = out_dir('linear') / 'summary.csv'
    key = lambda r: (r['method'], int(r['W']), int(r['H']), float(r['alpha']), int(r['seed']))
    if path.exists():
        new = {key(r) for r in rows}
        rows = [r for r in load_csv(path) if key(r) not in new] + rows
    save_csv(path, rows)

    if '10x20' in opt['boards']:
        base = {}
        for kind, name in ((simulate.RANDOM, 'random'), (simulate.GREEDY, 'greedy')):
            base[name] = simulate.monte_carlo(kind, 10, 20, 100_000, seed=7)
            print(name, base[name])
        save_json(out_dir('linear') / 'baselines.json', base)


if __name__ == '__main__':
    main()
