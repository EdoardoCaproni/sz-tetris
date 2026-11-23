"""The hand-made heuristic (tetris.heuristic_weights): one-step look-ahead
max_u [ r + J~(b') ] with a fixed J~ on the board features, the same rule
as linear TD but with weights chosen by hand instead of learned.

Width 6, heights 3 to 6: exact value, expected pieces and agreement with
the optimal placements (weighted by the visits of the heuristic itself).
10x20: Monte Carlo with `games` games (95% confidence interval).

Output: results/heuristic/summary.json
Run:    .venv\\Scripts\\python experiments\\exp_heuristic.py
"""
from common import args, out_dir, save_json

import dp
import linear
import simulate
from tetris import heuristic_weights


def main():
    opt = args(heights=[3, 4, 5, 6], games=100_000)
    out = {}
    for H in opt['heights']:
        t = dp.solve(6, H)
        nxt, rew = t['nxt'], t['rew']
        w = heuristic_weights(6, H)
        pol = linear.linear_policy_on_table(t, linear.table_features(t, 6, H), w)
        v = float(dp.evaluate_policy(nxt, rew, pol)[0])
        occ = dp.occupancy(nxt, pol)
        row = dict(value=v, ratio=v / float(t['Wstar'][0]),
                   pieces=float(dp.expected_pieces(nxt, rew, pol)[0]),
                   agreement=dp.agreement(t['Wstar'], nxt, rew, pol, occ),
                   weights=w.tolist())
        out[f'6x{H}'] = row
        print(f"6x{H}: {v:.4f} rows ({100 * row['ratio']:.1f}% of the optimum), "
              f"{row['pieces']:.2f} pieces, agreement {100 * row['agreement']:.1f}%")
    w = heuristic_weights(10, 20)
    r = simulate.monte_carlo(simulate.LINEAR, 10, 20, opt['games'], seed=7, weights=w)
    r['weights'] = w.tolist()
    out['10x20'] = r
    print(f"10x20: {r['mean']:.2f} +- {r['ci95']:.2f} rows, {r['pieces']:.1f} pieces, "
          f"capped {r['capped']}")
    save_json(out_dir('heuristic') / 'summary.json', out)


if __name__ == '__main__':
    main()
