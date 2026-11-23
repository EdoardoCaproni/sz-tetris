"""Exact solution of SZ-Tetris on width 6, heights 3 to 6.

For each height: number of boards, value iteration (sweeps, change per
sweep), the termination check (dp.safe_set), and the exact value of the
optimal policy and of the two baselines (random, greedy), with the
expected number of pieces placed.

Output: results/dp/summary.json (the tables are cached in results/dp_cache)
Run:    .venv\\Scripts\\python experiments\\exp_dp.py [heights=3,4,5,6]
"""
import time

from common import args, out_dir, save_json

import dp


def main():
    opt = args(width=6, heights=[3, 4, 5, 6])
    W = opt['width']
    summary = []
    for H in opt['heights']:
        t0 = time.perf_counter()
        t = dp.solve(W, H, verbose=True)
        nxt, rew = t['nxt'], t['rew']
        safe = dp.safe_set(nxt)
        row = dict(W=W, H=H, boards=int(t['keys'].size), states=2 * int(t['keys'].size),
                   sweeps=len(t['deltas']), deltas=[float(d) for d in t['deltas']],
                   safe_set=int(safe.size),
                   optimal=float(t['Wstar'][0]),
                   optimal_pieces=float(dp.expected_pieces(nxt, rew, t['pol'])[0]))
        for name in ('greedy', 'random'):
            probs = dp.baseline_probs(nxt, rew, name)
            row[name] = float(dp.evaluate_policy(nxt, rew, probs)[0])
            row[name + '_pieces'] = float(dp.expected_pieces(nxt, rew, probs)[0])
        row['seconds'] = time.perf_counter() - t0
        print(f"  H={H}: optimal {row['optimal']:.4f} rows, {row['optimal_pieces']:.2f} pieces; "
              f"greedy {row['greedy']:.4f}; random {row['random']:.4f}; safe set {row['safe_set']}")
        summary.append(row)
    save_json(out_dir('dp') / 'summary.json', summary)


if __name__ == '__main__':
    main()
