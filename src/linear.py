"""Linear function approximation (RLI ch. 9) and REINFORCE (RLI ch. 13).

Both methods play on the real board (no table), so they also work on
10x20, where the table of boards is impossible. The board is described by
the features of tetris.features (Bertsekas and Tsitsiklis): column
heights, height differences, maximum height, holes, constant.

Semi-gradient TD(0) on afterstates (RLI 9.3, 6.8)
  W_hat(b) = w . phi(b). When piece p is drawn on board b:
    target = max_a [ r_a + W_hat(b'_a) ]     (0 for a losing placement)
    w <- w + alpha [ target - w . phi(b) ] phi(b)
  "semi"-gradient: the target also depends on w, but it is treated as a
  constant (only the gradient of W_hat(b) = phi(b) is used).
  The move played is epsilon-greedy on r_a + W_hat(b'_a).

REINFORCE with baseline (RLI 13.3-13.4), gamma = 1
  policy  pi(a|s, theta) = exp(theta . x(s,a)) / sum_c exp(theta . x(s,c))
          x(s,a) = [ phi(b'_a), rows cleared by a, 1 if a loses ]
  baseline v_hat(s, w) = w . phi(b)
  after each episode, for t = 0 .. T-1:
    G_t = sum of the rewards after t
    delta = G_t - v_hat(S_t, w)
    w     <- w + alpha_w delta phi(b_t)
    theta <- theta + alpha_theta delta grad ln pi(A_t|S_t, theta)
  with grad ln pi(a|s) = x(s,a) - sum_c pi(c|s) x(s,c)   (softmax-linear)
"""
import time

import numpy as np

from tetris import njit, drop, dot, features, n_features, placements, unpack
from simulate import afterstates, softmax_x, softmax_probs
import dp
import simulate
import tabular


@njit(cache=True)
def td_linear(w, W, H, cells, max_steps, alpha, eps, max_len):
    """Semi-gradient TD(0) on afterstates for max_steps pieces.

    Episodes in progress are not carried over between calls: each call
    starts from the empty board. Returns (steps, episodes finished)."""
    A = cells.shape[1]
    nf = w.shape[0]
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    tmp = np.zeros(H, dtype=np.int64)
    phi = np.zeros((A, nf))
    phi_b = np.zeros(nf)
    rew = np.zeros(A)
    over = np.zeros(A, dtype=np.bool_)
    q = np.zeros(A)
    steps = 0
    episodes = 0
    while steps < max_steps:
        rows[:] = 0
        features(rows, W, H, phi_b)
        n = 0
        while n < max_len and steps < max_steps:
            steps += 1
            n += 1
            p = np.random.randint(2)
            afterstates(rows, W, H, cells[p], phi, rew, over, tmp)
            for a in range(A):
                q[a] = 0.0 if over[a] else rew[a] + dot(phi[a], w)
            target = q.max()
            w += alpha * (target - dot(phi_b, w)) * phi_b
            a = tabular.eps_greedy(q, eps)
            if over[a]:
                episodes += 1
                break
            drop(rows, H, full, cells[p, a])
            phi_b[:] = phi[a]
    return steps, episodes


@njit(cache=True)
def reinforce(theta, w, W, H, cells, max_steps, alpha_theta, alpha_w, max_len):
    """REINFORCE with baseline for about max_steps pieces (whole episodes).

    An episode longer than max_len pieces is cut there (its returns then
    miss the rows after the cut). Returns (steps, episodes)."""
    A = cells.shape[1]
    nf = w.shape[0]
    nx = theta.shape[0]
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    tmp = np.zeros(H, dtype=np.int64)
    phi = np.zeros((A, nf))
    rew = np.zeros(A)
    over = np.zeros(A, dtype=np.bool_)
    probs = np.zeros(A)
    grad = np.zeros(nx)
    # the episode: action features of every placement, choice, reward, phi(b_t)
    X = np.zeros((max_len, A, nx))
    acts = np.zeros(max_len, dtype=np.int64)
    rs = np.zeros(max_len)
    PB = np.zeros((max_len, nf))
    steps = 0
    episodes = 0
    while steps < max_steps:
        rows[:] = 0
        T = 0
        while T < max_len:
            features(rows, W, H, PB[T])
            p = np.random.randint(2)
            afterstates(rows, W, H, cells[p], phi, rew, over, tmp)
            softmax_x(phi, rew, over, X[T])
            softmax_probs(theta, X[T], probs)
            u = np.random.random()
            c = 0.0
            a = A - 1
            for k in range(A):
                c += probs[k]
                if u < c:
                    a = k
                    break
            acts[T] = a
            rs[T] = rew[a]
            T += 1
            if over[a]:
                break
            drop(rows, H, full, cells[p, a])
        steps += T
        episodes += 1
        G = 0.0
        for t in range(T - 1, -1, -1):
            G += rs[t]
            rs[t] = G                 # rs[t] now holds G_t
        for t in range(T):
            delta = rs[t] - dot(PB[t], w)
            w += alpha_w * delta * PB[t]
            softmax_probs(theta, X[t], probs)
            grad[:] = X[t, acts[t]]
            for k in range(A):
                grad -= probs[k] * X[t, k]
            theta += alpha_theta * delta * grad
    return steps, episodes


# --------------------------------------------- exact evaluation on tables

def table_features(table, W, H):
    """Features of every board of a dp table, shape (N, n_features)."""
    keys = table['keys']
    nf = n_features(W)
    out = np.zeros((keys.size, nf))
    _table_features(keys, W, H, out)
    return out


@njit(cache=True)
def _table_features(keys, W, H, out):
    rows = np.zeros(H, dtype=np.int64)
    for i in range(keys.shape[0]):
        unpack(keys[i], W, H, rows)
        features(rows, W, H, out[i])


def linear_policy_on_table(table, Phi, w):
    """Greedy policy of W_hat = Phi w on the table (same rule as simulate.LINEAR)."""
    nxt, rew = table['nxt'], table['rew']
    q = np.where(nxt >= 0, rew + (Phi @ w)[np.maximum(nxt, 0)], simulate.LOSE)
    return q.argmax(axis=2).astype(np.int8)


def softmax_probs_on_table(table, Phi, theta):
    """pi(a | b, p) for every board and piece of the table."""
    nxt, rew = table['nxt'], table['rew']
    nf = Phi.shape[1]
    h = np.where(nxt >= 0,
                 (Phi @ theta[:nf])[np.maximum(nxt, 0)] + theta[nf] * rew,
                 theta[nf + 1])
    h = h - h.max(axis=2, keepdims=True)
    e = np.exp(h)
    return e / e.sum(axis=2, keepdims=True)


# ---------------------------------------------------------------- training

def train(method, W, H, steps, alpha, eps=0.0, alpha_w=None, seed_=0,
          n_checkpoints=30, table=None, eval_games=200, max_len=None,
          first=10_000):
    """Train td_linear or reinforce for `steps` pieces, with checkpoints.

    At each checkpoint the policy is evaluated exactly if a dp table is
    given (small boards), otherwise by Monte Carlo with eval_games games.
    Returns (curve, final, learned)."""
    cells = placements(W)[0]
    nf = n_features(W)
    if max_len is None:
        max_len = 100_000 if method == 'td_linear' else 20_000
    tabular.seed(seed_)
    w = np.zeros(nf)
    theta = np.zeros(nf + 2)
    Phi = table_features(table, W, H) if table is not None else None
    curve = []
    done = 0
    episodes = 0
    t_train = 0.0
    for k, ck in enumerate(np.unique(np.geomspace(first, steps, n_checkpoints).astype(np.int64))):
        t0 = time.perf_counter()
        if method == 'td_linear':
            s, e = td_linear(w, W, H, cells, int(ck - done), alpha, eps, max_len)
        elif method == 'reinforce':
            s, e = reinforce(theta, w, W, H, cells, int(ck - done), alpha,
                             alpha_w if alpha_w is not None else alpha, max_len)
        else:
            raise ValueError(method)
        t_train += time.perf_counter() - t0
        done += s
        episodes += e
        row = dict(steps=done, episodes=episodes)
        row.update(evaluate(method, W, H, w, theta, table, Phi, eval_games,
                            seed=10_000 + k))
        # the test games reseed the random generator (the same test pieces
        # for every run): training goes on from a seed of its own, so that
        # different seeds stay independent runs (exact evaluation on a
        # table uses no random numbers)
        if table is None:
            tabular.seed(1_000_003 * (seed_ + 1) + k)
        if not np.isfinite(w).all() or not np.isfinite(theta).all():
            row['diverged'] = True
            curve.append(row)
            break
        curve.append(row)
    final = dict(method=method, W=W, H=H, steps=done, episodes=episodes,
                 train_seconds=t_train, **{k: v for k, v in curve[-1].items()
                                           if k not in ('steps', 'episodes')})
    learned = dict(w=w.copy(), theta=theta.copy())
    return curve, final, learned


def evaluate(method, W, H, w, theta, table=None, Phi=None, games=200, seed=0):
    """Expected rows of the learned policy: exact on a table, else Monte Carlo."""
    if not (np.isfinite(w).all() and np.isfinite(theta).all()):
        return dict(value=float('nan'))
    if table is not None:
        nxt, rew = table['nxt'], table['rew']
        if method == 'td_linear':
            pol = linear_policy_on_table(table, Phi, w)
            v = dp.evaluate_policy(nxt, rew, pol, tol=1e-8)[0]
        else:
            v = dp.evaluate_policy(nxt, rew, softmax_probs_on_table(table, Phi, theta), tol=1e-8)[0]
        return dict(value=float(v), ratio=float(v / table['Wstar'][0]))
    if method == 'td_linear':
        r = simulate.monte_carlo(simulate.LINEAR, W, H, games, seed, weights=w)
    else:
        r = simulate.monte_carlo(simulate.SOFTMAX, W, H, games, seed, weights=theta)
    return dict(value=r['mean'], ci95=r['ci95'], pieces=r['pieces'], capped=r['capped'])
