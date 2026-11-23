"""The five tabular methods on 10x20, with a hash table of the boards met
(src/tabular_hash.py).

Each method uses its best setting of the 6x5 grid (the one of the final
runs, read from results/tabular/final), `seeds` seeds, `steps` pieces.
At each checkpoint the greedy policy is tested with 200 games; the final
policy with `final_games` games. Recorded: rows (95% CI), boards in the
table, and the share of the boards met in the test games that were in
the table (coverage).

Output: results/tabular_10x20/<method>_s<seed>/{config.json,train.csv,final.json}
        results/tabular_10x20/summary.csv
Run:    .venv\\Scripts\\python experiments\\exp_tabular_10x20.py [steps=10000000 seeds=5]
"""
from common import args, out_dir, run_grid, save_csv, save_json, load_json, RESULTS

import tabular
import tabular_hash


def best_setting(method):
    c = load_json(RESULTS / 'tabular' / 'final' / f'{method}_H5_s0' / 'config.json')
    return c['alpha'], c['eps'], c['q0']


def run_one(cfg):
    curve, final, _ = tabular_hash.train(
        cfg['method'], 10, 20, cfg['steps'], alpha=cfg['alpha'], eps=cfg['eps'],
        q0=cfg['q0'], seed_=cfg['seed'], final_games=cfg['final_games'])
    final.update({k: v for k, v in cfg.items() if k != 'save'})
    d = out_dir(*cfg['save'])
    save_json(d / 'config.json', cfg)
    save_csv(d / 'train.csv', curve)
    save_json(d / 'final.json', final)
    print(f"{cfg['method']:15s} s{cfg['seed']}: {final['value']:.2f} +- {final['ci95']:.2f} rows, "
          f"{final['boards']:,} boards, coverage {100 * final['coverage']:.0f}%", flush=True)
    return final


def main():
    opt = args(steps=10_000_000, seeds=5, final_games=2000, processes=6)
    cfgs = []
    for m in tabular.METHODS:
        a, e, q0 = best_setting(m)
        for s in range(opt['seeds']):
            cfgs.append(dict(method=m, alpha=a, eps=e, q0=q0, steps=opt['steps'], seed=s,
                             final_games=opt['final_games'],
                             save=['tabular_10x20', f'{m}_s{s}']))
    # the table of a Q method is 34 values per board: few runs at a time
    rows = run_grid(run_one, cfgs, processes=opt['processes'])
    save_csv(out_dir('tabular_10x20') / 'summary.csv', rows)


if __name__ == '__main__':
    main()
