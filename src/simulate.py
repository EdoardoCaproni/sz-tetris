"""Playing games on any board (no table), and Monte Carlo estimates.

On the small boards the value of a policy is computed exactly (dp.py). On
10x20 the table is impossible, so the expected number of rows is
estimated by playing n games (deck 04, Monte Carlo):

  mean = (1/n) sum_k rows_k,   95% confidence interval mean +- 1.96 s / sqrt(n)

with s the sample standard deviation (law of large numbers; the error
shrinks as 1/sqrt(n)).

Policies (same rules as the page, viewer/engine.js):
  RANDOM   every placement with the same probability
  GREEDY   the greedy policy of deck 09: most rows cleared now, a losing
           placement only if all lose, ties at random
  LINEAR   afterstate values from features: max r + w.phi(b'), a losing
           placement is worth 0 and loses ties, remaining ties to the
           first placement
  SOFTMAX  REINFORCE policy: placement a with probability
           exp(theta.x_a) / sum_c exp(theta.x_c)
  TABLE    afterstate values from a table (keys sorted, values), as LINEAR
"""
import numpy as np

from tetris import njit, drop, features, pack, dot, GAME_OVER, n_features, placements

RANDOM, GREEDY, LINEAR, SOFTMAX, TABLE = 0, 1, 2, 3, 4
LOSE = -1e-12         # value of a losing placement when choosing greedily


@njit(cache=True)
def afterstates(rows, W, H, cells_p, phi, rew, over, tmp):
    """Features phi[a], rows cleared rew[a] and game over flag of every
    placement of the current piece (cells_p = cells[piece])."""
    full = (1 << W) - 1
    for a in range(cells_p.shape[0]):
        tmp[:] = rows
        c = drop(tmp, H, full, cells_p[a])
        if c == GAME_OVER:
            over[a] = True
            rew[a] = 0.0
            phi[a, :] = 0.0
        else:
            over[a] = False
            rew[a] = c
            features(tmp, W, H, phi[a])


@njit(cache=True)
def softmax_x(phi, rew, over, x):
    """REINFORCE action features x(s,a) = [phi(b'_a), rows cleared, game over]."""
    nf = phi.shape[1]
    for a in range(phi.shape[0]):
        x[a, :nf] = phi[a]
        x[a, nf] = rew[a]
        x[a, nf + 1] = 1.0 if over[a] else 0.0


@njit(cache=True)
def softmax_probs(theta, x, out):
    A = x.shape[0]
    hmax = -1e300
    for a in range(A):
        out[a] = dot(x[a], theta)
        if out[a] > hmax:
            hmax = out[a]
    s = 0.0
    for a in range(A):
        out[a] = np.exp(out[a] - hmax)
        s += out[a]
    for a in range(A):
        out[a] /= s


@njit(cache=True)
def _choose(kind, rows, W, H, cells_p, phi, rew, over, tmp, q, x, probs,
            weights, keys, values):
    A = cells_p.shape[0]
    afterstates(rows, W, H, cells_p, phi, rew, over, tmp)
    if kind == RANDOM:
        return np.random.randint(A)
    if kind == GREEDY:
        for a in range(A):
            q[a] = -1000.0 if over[a] else rew[a]
        best = q.max()
        n = 0
        for a in range(A):
            if q[a] == best:
                n += 1
        k = np.random.randint(n)
        for a in range(A):
            if q[a] == best:
                if k == 0:
                    return a
                k -= 1
        return 0
    if kind == SOFTMAX:
        softmax_x(phi, rew, over, x)
        softmax_probs(weights, x, probs)
        u = np.random.random()
        c = 0.0
        for a in range(A):
            c += probs[a]
            if u < c:
                return a
        return A - 1
    # LINEAR or TABLE: greedy on r + value of the afterstate
    full = (1 << W) - 1
    for a in range(A):
        if over[a]:
            q[a] = LOSE
        elif kind == LINEAR:
            q[a] = rew[a] + dot(phi[a], weights)
        else:
            tmp[:] = rows
            drop(tmp, H, full, cells_p[a])
            q[a] = rew[a] + values[np.searchsorted(keys, pack(tmp, W))]
    return np.argmax(q)


@njit(cache=True)
def play_games(kind, W, H, cells, games, max_pieces, nf, weights, keys, values):
    """Play games from the empty board; returns rows and pieces per game.

    A game is stopped after max_pieces pieces (counted in the result, so
    a capped game is visible as pieces == max_pieces)."""
    A = cells.shape[1]
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    tmp = np.zeros(H, dtype=np.int64)
    phi = np.zeros((A, nf))
    rew = np.zeros(A)
    over = np.zeros(A, dtype=np.bool_)
    q = np.zeros(A)
    x = np.zeros((A, nf + 2))
    probs = np.zeros(A)
    out_rows = np.zeros(games, dtype=np.int64)
    out_pieces = np.zeros(games, dtype=np.int64)
    for g in range(games):
        rows[:] = 0
        total = 0
        n = 0
        while n < max_pieces:
            p = np.random.randint(2)
            a = _choose(kind, rows, W, H, cells[p], phi, rew, over, tmp, q, x,
                        probs, weights, keys, values)
            c = drop(rows, H, full, cells[p, a])
            if c == GAME_OVER:
                break
            total += c
            n += 1
        out_rows[g] = total
        out_pieces[g] = n
    return out_rows, out_pieces


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


def monte_carlo(kind, W, H, games, seed=0, weights=None, keys=None,
                values=None, max_pieces=10**7):
    """Estimate the expected rows of a policy by playing games.

    Returns dict(mean, ci95, std, games, pieces, capped)."""
    cells = placements(W)[0]
    nf = n_features(W)
    if weights is None:
        weights = np.zeros(nf + 2)
    if keys is None:
        keys = np.zeros(1, dtype=np.int64)
        values = np.zeros(1)
    _seed(seed)
    rows, pieces = play_games(kind, W, H, cells, games, max_pieces, nf,
                              np.asarray(weights, dtype=np.float64), keys,
                              np.asarray(values, dtype=np.float64))
    s = rows.std(ddof=1) if games > 1 else 0.0
    return dict(mean=float(rows.mean()), ci95=float(1.96 * s / np.sqrt(games)),
                std=float(s), games=games, pieces=float(pieces.mean()),
                capped=int((pieces >= max_pieces).sum()))
