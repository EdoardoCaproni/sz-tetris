"""Linear function approximation and REINFORCE: 6x5 and 6x6 (exact
evaluation, so the loss due to the features is measured exactly) and
10x20 (Monte Carlo evaluation), plus the baselines on 10x20.

Grid, the same on every board:
  td_linear  alpha in {0.001, 0.003, 0.01}, epsilon = 0.01
  reinforce  alpha_theta in {0.001, 0.003, 0.01, 0.03, 0.1}, alpha_w = 0.01
(the last two REINFORCE values were added after 0.01 came out best; run
them alone with methods=reinforce alphas=0.03,0.1: the summary keeps the
runs already there).
Each setting with `seeds` seeds; budgets in pieces placed (steps), fixed
in advance: 2e7 on 6x5 and 6x6, 2.5e8 on 10x20 (first run: 5e7, where
REINFORCE was still improving).
Checkpoints: exact on the small boards; on 10x20 Monte Carlo with
`ck_games` games (95% CI about +-2 rows at 90 rows). The final policy of
every 10x20 run is re-evaluated with `final_games` games.

Stopping rule (plateau check, printed and saved in plateau.json): for
each seed, d = mean of the last 3 checkpoints - mean of the 3 before
(about 1.2e8-2.5e8 against 4.4e7-8.8e7 pieces on 10x20). A setting is
still improving if the 95% interval of the mean of d over the seeds
(t distribution, n - 1 degrees of freedom) lies above 0; otherwise it
has levelled off. Averaging checkpoints and pairing by seed matter: with
a constant step size the greedy policy of linear TD moves between
checkpoints by much more than the Monte Carlo error of one checkpoint,
so comparing two single checkpoints (the first version of this check)
calls noise "still improving". A seed whose last checkpoint is below
1 row is reported as collapsed.

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
STEPS = {'6x5': 20_000_000, '6x6': 20_000_000, '10x20': 250_000_000}


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
               ck_games=1000, methods=['td_linear', 'reinforce'], alphas=[0.0],
               stage='run')
    if opt['stage'] == 'plateau':
        return plateau()
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
                                     steps=STEPS[b], seed=s,
                                     eval_games=opt['ck_games'] if W * H > 36 else 200,
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
        plateau()


# 97.5% quantile of Student's t with df degrees of freedom (no scipy)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228}


def t975(df):
    return T975.get(df, 1.96)


def plateau(board='10x20', k=3):
    """The stopping rule of the docstring, for every (method, alpha) on board."""
    import glob
    out = []
    for m, alphas in ALPHAS.items():
        for a in alphas:
            curves = [load_csv(f) for f in sorted(glob.glob(str(
                out_dir('linear') / f'{m}_{board}_a{a}_s*' / 'train.csv')))]
            if not curves:
                continue
            n = min(len(c) for c in curves)
            steps = np.array([float(r['steps']) for r in curves[0][:n]])
            v = np.array([[float(r['value']) for r in c[:n]] for c in curves])
            last, prev = v[:, -k:].mean(1), v[:, -2 * k:-k].mean(1)
            d = last - prev
            ci = t975(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            row = dict(method=m, alpha=a, seeds=len(curves),
                       steps_prev=[steps[-2 * k], steps[-k - 1]], steps_last=[steps[-k], steps[-1]],
                       mean_prev=float(prev.mean()), mean_last=float(last.mean()),
                       diff=float(d.mean()), ci95=float(ci),
                       improving=bool(d.mean() - ci > 0),
                       collapsed=int((v[:, -1] < 1).sum()))
            out.append(row)
            print(f"{m:10s} a={a:<6} {row['mean_prev']:6.1f} -> {row['mean_last']:6.1f}  "
                  f"d = {row['diff']:+5.1f} +- {ci:4.1f}  "
                  f"{'still improving' if row['improving'] else 'levelled off'}"
                  + (f"  ({row['collapsed']} seeds collapsed)" if row['collapsed'] else ''))
    save_json(out_dir('linear') / 'plateau.json', out)
    return out


if __name__ == '__main__':
    main()
