"""Tabular RL on width 6: a small grid on 6x5, then the best settings on
heights 4, 5, 6 with the same budget.

Stage "grid" (6x5, `grid_episodes` episodes, `grid_seeds` seeds):
  alpha in {0.03, 0.1, 0.3, 0.6} x epsilon in {0.01, 0.05, 0.1, 0.2}, q0 = 0,
  plus the optimistic variant: epsilon = 0 and q0 = 2H (above the values
  seen; exploration comes only from the optimistic start).
  Monte Carlo control has no alpha (averages of returns).
Stage "final" (heights 4, 5, 6, `episodes` episodes, `seeds` seeds): the
  setting with the best mean final value in the grid, for each method.

Every run evaluates the greedy policy of its table exactly (dp.py) at 40
checkpoints, so the learning curves have no simulation noise.

Output: results/tabular/grid/summary.csv
        results/tabular/final/<method>_H<H>_s<seed>/{config.json,train.csv,final.json,learned.npz}
        results/tabular/final/summary.csv
Run:    .venv\\Scripts\\python experiments\\exp_tabular.py [stage=grid,final]
"""
import numpy as np

from common import args, out_dir, run_grid, save_csv, save_json, load_csv

import dp
import tabular

ALPHAS = (0.03, 0.1, 0.3, 0.6)
EPSILONS = (0.01, 0.05, 0.1, 0.2)


def grid_configs(H):
    out = []
    for m in tabular.METHODS:
        alphas = (0.0,) if m == 'mc' else ALPHAS
        for a in alphas:
            for e in EPSILONS:
                out.append(dict(method=m, alpha=a, eps=e, q0=0.0))
            out.append(dict(method=m, alpha=a, eps=0.0, q0=2.0 * H))
    return out


def run_one(cfg):
    table = dp.solve(6, cfg['H'])
    occ = dp.occupancy(table['nxt'], table['pol'])
    curve, final, learned = tabular.train(
        cfg['method'], table, cfg['episodes'], cfg['alpha'], cfg['eps'],
        cfg['q0'], cfg['seed'], occ_star=occ)
    final.update(cfg)
    if cfg.get('save'):
        d = out_dir(*cfg['save'])
        save_json(d / 'config.json', cfg)
        save_csv(d / 'train.csv', curve)
        save_json(d / 'final.json', final)
        np.savez_compressed(d / 'learned.npz', **learned)
    return final


def main():
    opt = args(stage=['grid', 'final'], grid_episodes=2_000_000, grid_seeds=3,
               episodes=10_000_000, seeds=5, heights=[4, 5, 6])
    for H in opt['heights']:
        dp.solve(6, H, verbose=True)          # build the caches before forking

    if 'grid' in opt['stage']:
        cfgs = [dict(c, H=5, episodes=opt['grid_episodes'], seed=s)
                for c in grid_configs(5) for s in range(opt['grid_seeds'])]
        rows = run_grid(run_one, cfgs)
        save_csv(out_dir('tabular', 'grid') / 'summary.csv', rows)

    if 'final' in opt['stage']:
        grid = load_csv(out_dir('tabular', 'grid') / 'summary.csv')
        best = {}
        for m in tabular.METHODS:
            scores = {}
            for r in grid:
                if r['method'] == m:
                    key = (float(r['alpha']), float(r['eps']), float(r['q0']) > 0)
                    scores.setdefault(key, []).append(float(r['value']))
            key = max(scores, key=lambda k: np.mean(scores[k]))
            best[m] = key
            print(f'{m}: alpha={key[0]} eps={key[1]} optimistic={key[2]} '
                  f'(grid mean {np.mean(scores[key]):.3f})')
        cfgs = []
        for H in opt['heights']:
            for m in tabular.METHODS:
                a, e, optimistic = best[m]
                for s in range(opt['seeds']):
                    cfgs.append(dict(method=m, H=H, alpha=a, eps=e,
                                     q0=2.0 * H if optimistic else 0.0,
                                     episodes=opt['episodes'], seed=s,
                                     save=['tabular', 'final', f'{m}_H{H}_s{s}']))
        # 6x6 tables are large (a few hundred MB per run): fewer processes
        small = [c for c in cfgs if c['H'] < 6]
        large = [c for c in cfgs if c['H'] == 6]
        rows = run_grid(run_one, small) + (run_grid(run_one, large, processes=8) if large else [])
        for r in rows:
            r.pop('save', None)
        save_csv(out_dir('tabular', 'final') / 'summary.csv', rows)


if __name__ == '__main__':
    main()
