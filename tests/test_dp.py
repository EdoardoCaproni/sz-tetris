"""Exact DP: table, value iteration, termination, policy evaluation,
and the Monte Carlo cross-checks."""
from functools import lru_cache

import numpy as np
import pytest

import dp
import simulate


@lru_cache(None)
def table(H):
    keys, nxt, rew = dp.build_table(6, H)
    Wstar, deltas = dp.value_iteration(nxt, rew)
    pol = dp.greedy_from_W(Wstar, nxt, rew)
    return dict(keys=keys, nxt=nxt, rew=rew, Wstar=Wstar, deltas=deltas, pol=pol)


@pytest.mark.parametrize('H, boards, value', [
    (3, 330, 2.3046875),          # first feasibility script: 660 states, 2.30
    (4, 6_349, 4.328971164),      # 12,698 states, 4.33
    (5, 91_658, 7.426176286),     # 183,316 states, 7.43
])
def test_known_values(H, boards, value):
    t = table(H)
    assert t['keys'].size == boards
    assert t['keys'][0] == 0
    assert abs(t['Wstar'][0] - value) < 1e-8


@pytest.mark.parametrize('H', [3, 4, 5])
def test_bellman_fixed_point(H):
    t = table(H)
    out = np.zeros_like(t['Wstar'])
    residual = dp._bellman(t['Wstar'], t['nxt'], t['rew'], out)
    assert residual < 1e-9
    assert (t['Wstar'] >= 0).all()


@pytest.mark.parametrize('H', [3, 4, 5])
def test_every_policy_terminates(H):
    assert dp.safe_set(table(H)['nxt']).size == 0


def test_safe_set_detects_an_endless_game():
    # a made-up table where board 0 can always go to board 1 and back
    nxt = np.array([[[1, -1], [1, -1]], [[0, -1], [-1, 0]]], dtype=np.int32)
    assert list(dp.safe_set(nxt)) == [0, 1]
    nxt[1, 1] = -1                     # now the Z on board 1 always loses
    assert dp.safe_set(nxt).size == 0


@pytest.mark.parametrize('H', [3, 4])
def test_policy_evaluation_of_the_optimal_policy(H):
    t = table(H)
    V = dp.evaluate_policy(t['nxt'], t['rew'], t['pol'])
    assert np.allclose(V, t['Wstar'], atol=1e-8)
    # the same policy written as probabilities
    probs = np.zeros(t['nxt'].shape)
    np.put_along_axis(probs, t['pol'].astype(np.int64)[:, :, None], 1.0, axis=2)
    V2 = dp.evaluate_policy(t['nxt'], t['rew'], probs)
    assert np.allclose(V2, V, atol=1e-8)


def test_optimal_is_better_than_any_other_policy():
    t = table(4)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pol = rng.integers(0, t['nxt'].shape[2], t['pol'].shape).astype(np.int8)
        V = dp.evaluate_policy(t['nxt'], t['rew'], pol)
        assert (V <= t['Wstar'] + 1e-8).all()


def test_occupancy_counts_the_pieces():
    # visits to boards = pieces drawn = pieces placed + the one that loses
    t = table(4)
    occ = dp.occupancy(t['nxt'], t['pol'])
    L = dp.expected_pieces(t['nxt'], t['rew'], t['pol'])
    assert abs(occ.sum() - (L[0] + 1)) < 1e-6


def test_agreement_of_the_optimal_policy_is_one():
    t = table(4)
    occ = dp.occupancy(t['nxt'], t['pol'])
    assert dp.agreement(t['Wstar'], t['nxt'], t['rew'], t['pol'], occ) == 1.0


@pytest.mark.parametrize('kind, name', [(simulate.RANDOM, 'random'),
                                        (simulate.GREEDY, 'greedy'),
                                        (simulate.TABLE, 'optimal')])
def test_monte_carlo_agrees_with_exact(kind, name):
    t = table(4)
    if name == 'optimal':
        exact = t['Wstar'][0]
    else:
        exact = dp.evaluate_policy(t['nxt'], t['rew'],
                                   dp.baseline_probs(t['nxt'], t['rew'], name))[0]
    r = simulate.monte_carlo(kind, 6, 4, 20_000, seed=3, keys=t['keys'], values=t['Wstar'])
    sigma = r['std'] / np.sqrt(r['games'])
    assert abs(r['mean'] - exact) < 4 * sigma + 1e-12
