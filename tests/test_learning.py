"""Smoke tests of every learning method on a tiny board (6x3, 330 boards):
each one must get close to the exact optimum, measured exactly."""
import numpy as np
import pytest

import dp
import linear
import tabular

TABLE = None


def table():
    global TABLE
    if TABLE is None:
        keys, nxt, rew = dp.build_table(6, 3)
        Wstar, deltas = dp.value_iteration(nxt, rew)
        TABLE = dict(keys=keys, nxt=nxt, rew=rew, Wstar=Wstar, deltas=deltas,
                     pol=dp.greedy_from_W(Wstar, nxt, rew))
    return TABLE


@pytest.mark.parametrize('method', tabular.METHODS)
def test_tabular_learns_the_tiny_board(method):
    t = table()
    curve, final, _ = tabular.train(method, t, 200_000, alpha=0.1, eps=0.05,
                                    seed_=1, n_checkpoints=5)
    assert final['ratio'] > 0.95, final
    assert curve[-1]['value'] <= t['Wstar'][0] + 1e-9


def test_td_afterstate_values_approach_the_exact_ones():
    t = table()
    occ = dp.occupancy(t['nxt'], t['pol'])
    curve, _, learned = tabular.train('td_afterstate', t, 200_000, alpha=0.05,
                                      eps=0.05, seed_=2, n_checkpoints=5,
                                      occ_star=occ)
    assert curve[-1]['w_rmse'] < curve[0]['w_rmse']
    assert curve[-1]['w_rmse'] < 0.2


@pytest.mark.parametrize('method, alpha', [('td_linear', 0.01), ('reinforce', 0.01)])
def test_linear_methods_learn(method, alpha):
    t = table()
    random = dp.evaluate_policy(t['nxt'], t['rew'],
                                dp.baseline_probs(t['nxt'], t['rew'], 'random'))[0]
    curve, final, learned = linear.train(method, 6, 3, 300_000, alpha, eps=0.01,
                                         alpha_w=0.01, seed_=0, table=t,
                                         n_checkpoints=4)
    assert np.isfinite(learned['w']).all()
    # the features lose information (on 6x3 linear TD stays below half of
    # the optimum), so this only checks that learning happens
    assert final['value'] > 10 * random
    assert final['ratio'] > 0.3, final


def _policy_from_hash(t, method, V, d, W, H):
    """Greedy policy of a hash-table learner on every board of the dp table."""
    import tabular_hash
    from tetris import unpack
    nxt, rew, keys = t['nxt'], t['rew'], t['keys']
    pol = np.zeros((keys.size, 2), dtype=np.int8)
    rows = np.zeros(H, dtype=np.int64)
    for i, k in enumerate(keys):
        unpack(k, W, H, rows)
        b = tabular_hash.lookup(d, rows, W, H)
        for p in range(2):
            if method != 'td_afterstate':
                pol[i, p] = np.argmax(V[b, p]) if b >= 0 else 0
                continue
            q = np.full(nxt.shape[2], tabular_hash.LOSE)
            for a in range(nxt.shape[2]):
                j = nxt[i, p, a]
                if j >= 0:
                    unpack(keys[j], W, H, rows)
                    jj = tabular_hash.lookup(d, rows, W, H)
                    q[a] = rew[i, p, a] + (V[jj] if jj >= 0 else 0.0)
            pol[i, p] = np.argmax(q)
    return pol


@pytest.mark.parametrize('method', tabular.METHODS)
def test_hash_table_learners_learn_the_tiny_board(method):
    """The 10x20 versions (boards in a hash table, moves on the real board)
    are the same algorithms: on 6x3 they also get close to the optimum,
    measured exactly on the dp table."""
    import tabular_hash
    t = table()
    _, final, learned = tabular_hash.train(method, 6, 3, 1_000_000, alpha=0.1, eps=0.05,
                                           seed_=1, n_checkpoints=3, eval_games=10,
                                           final_games=10, capacity=1000)
    assert final['boards'] <= t['keys'].size
    pol = _policy_from_hash(t, method, learned['V'], learned['d'], 6, 3)
    v = dp.evaluate_policy(t['nxt'], t['rew'], pol)[0]
    assert v > 0.9 * t['Wstar'][0], (method, v)


def test_heuristic_beats_greedy():
    from tetris import heuristic_weights
    t = table()
    pol = linear.linear_policy_on_table(t, linear.table_features(t, 6, 3), heuristic_weights(6, 3))
    v = dp.evaluate_policy(t['nxt'], t['rew'], pol)[0]
    greedy = dp.evaluate_policy(t['nxt'], t['rew'], dp.baseline_probs(t['nxt'], t['rew'], 'greedy'))[0]
    assert v > greedy, (v, greedy)
