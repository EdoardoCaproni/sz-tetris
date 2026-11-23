"""Exact dynamic programming for SZ-Tetris on small boards.

The MDP (professor's template, deck 07):
  state      x_t = (b_t, p_t): board and current piece
  input      u_t = a placement of p_t (orientation and column)
  dynamics   b_{t+1} = f(b_t, p_t, u_t)  (drop + clear, src/tetris.py)
  uncertainty w_t = p_{t+1}, S or Z with probability 1/2
  reward     r(b_t, p_t, u_t) = rows cleared; game over is a termination
             state, absorbing and with no reward
  objective  max E[ sum of rewards until game over ]  (total reward, no
             discount: deck 11, infinite horizon with a termination state)

Afterstates (RLI 6.8; "post-decision states" in Bertsekas' table). The
piece is drawn after the move, so the value can be defined on the board
alone, before the next piece is known:

  W(b) = 1/2 * sum_p max_u [ r(b,p,u) + W(f(b,p,u)) ],   W(game over) = 0

and the value of a state is V(b, p) = max_u [ r(b,p,u) + W(f(b,p,u)) ].
One table of boards instead of boards x pieces, and the expectation is
taken before the max, over the next piece only.

Everything works on the table of the boards reachable from the empty one:
  keys[i]       packed board i (sorted, keys[0] = 0 = empty board)
  nxt[i, p, a]  index of the board after placement a of piece p, or -1
                if the placement ends the game
  rew[i, p, a]  rows cleared by it
The learners in tabular.py use the same table only as a fast simulator:
they draw a piece, pick a placement and read the next board and reward,
never the probabilities or the values.
"""
import time
from pathlib import Path

import numpy as np

from tetris import njit, placements, pack, unpack, drop, GAME_OVER

CACHE = Path(__file__).resolve().parent.parent / 'results' / 'dp_cache'


# ------------------------------------------------------------ the table

@njit(cache=True)
def _expand(keys, W, H, cells):
    """Successor key (or -1) and reward of every placement of every board."""
    n = keys.shape[0]
    A = cells.shape[1]
    full = (1 << W) - 1
    out = np.empty((n, 2, A), dtype=np.int64)
    rew = np.zeros((n, 2, A), dtype=np.int8)
    rows = np.zeros(H, dtype=np.int64)
    for i in range(n):
        for p in range(2):
            for a in range(A):
                unpack(keys[i], W, H, rows)
                c = drop(rows, H, full, cells[p, a])
                if c == GAME_OVER:
                    out[i, p, a] = -1
                else:
                    out[i, p, a] = pack(rows, W)
                    rew[i, p, a] = c
    return out, rew


def build_table(W, H, verbose=False):
    """Breadth-first search of the boards reachable from the empty board."""
    assert W * H <= 63
    cells = placements(W)[0]
    t0 = time.perf_counter()
    known = np.zeros(1, dtype=np.int64)
    frontier = known
    while frontier.size:
        succ, _ = _expand(frontier, W, H, cells)
        cand = np.unique(succ[succ >= 0])
        frontier = np.setdiff1d(cand, known, assume_unique=True)
        known = np.union1d(known, frontier)
    succ, rew = _expand(known, W, H, cells)
    nxt = np.where(succ >= 0, np.searchsorted(known, succ), -1).astype(np.int32)
    if verbose:
        print(f'{W}x{H}: {known.size:,} boards, {time.perf_counter() - t0:.1f} s')
    return known, nxt, rew


# ------------------------------------------------------- value iteration

@njit(cache=True)
def _bellman(Wv, nxt, rew, out):
    """One sweep of the afterstate Bellman operator; returns the sup-norm change."""
    N, _, A = nxt.shape
    delta = 0.0
    for i in range(N):
        total = 0.0
        for p in range(2):
            best = 0.0          # every placement is allowed; losing is worth 0
            for a in range(A):
                j = nxt[i, p, a]
                if j >= 0:
                    v = rew[i, p, a] + Wv[j]
                    if v > best:
                        best = v
            total += best
        out[i] = 0.5 * total
        d = abs(out[i] - Wv[i])
        if d > delta:
            delta = d
    return delta


def value_iteration(nxt, rew, tol=1e-10, max_sweeps=100_000):
    """W^(0) = 0, W^(i+1) = T W^(i) until ||W^(i+1) - W^(i)||_inf < tol.

    Returns W and the list of sup-norm changes, one per sweep. Total
    reward with a termination state (deck 11): convergence needs every
    policy to reach game over with probability 1, see safe_set.
    """
    N = nxt.shape[0]
    Wv = np.zeros(N)
    new = np.zeros(N)
    deltas = []
    for _ in range(max_sweeps):
        d = _bellman(Wv, nxt, rew, new)
        Wv, new = new, Wv
        deltas.append(d)
        if d < tol:
            break
    return Wv, np.array(deltas)


@njit(cache=True)
def q_values(Wv, nxt, rew, i, p):
    """r + W(b') for every placement of piece p on board i (0 if it loses)."""
    A = nxt.shape[2]
    q = np.zeros(A)
    for a in range(A):
        j = nxt[i, p, a]
        if j >= 0:
            q[a] = rew[i, p, a] + Wv[j]
    return q


@njit(cache=True)
def greedy_from_W(Wv, nxt, rew):
    """Policy that maximises r + W(b'), ties to the first placement.

    A losing placement is worth 0, like a placement that does not lose
    but leads to a board of value 0; on that tie the one that does not
    lose is preferred (same value, one more piece on the board)."""
    N, _, A = nxt.shape
    pol = np.zeros((N, 2), dtype=np.int8)
    for i in range(N):
        for p in range(2):
            best = -1.0
            for a in range(A):
                j = nxt[i, p, a]
                v = rew[i, p, a] + Wv[j] if j >= 0 else -1e-12
                if v > best:
                    best = v
                    pol[i, p] = a
    return pol


# --------------------------------------------- termination (A2 and A3)

def safe_set(nxt):
    """Boards from which some policy can survive forever, whatever pieces come.

    Greatest set S such that, for every board in S and every piece, some
    placement does not lose and leads back into S. Start from all boards
    and remove the ones that break the rule until nothing changes.
    If S is empty, from every board some finite sequence of pieces ends
    the game whatever the player does; that sequence has positive
    probability, so every policy reaches game over with probability 1
    (assumption A3 of the total-cost setting, deck 11).
    """
    N = nxt.shape[0]
    alive = np.ones(N + 1, dtype=bool)
    alive[N] = False                    # slot N = game over
    idx = np.where(nxt >= 0, nxt, N)
    while True:
        new = np.append(alive[idx].any(axis=2).all(axis=1), False)
        if (new == alive).all():
            return np.flatnonzero(alive[:N])
        alive = new


# ------------------------------------------------- evaluating a policy

@njit(cache=True)
def _eval_sweep(V, nxt, rew, probs_or_pol, deterministic, out, count_pieces):
    N, _, A = nxt.shape
    delta = 0.0
    for i in range(N):
        total = 0.0
        for p in range(2):
            if deterministic:
                a = probs_or_pol[i, p, 0]
                j = nxt[i, p, int(a)]
                if j >= 0:
                    total += (1.0 if count_pieces else rew[i, p, int(a)]) + V[j]
            else:
                for a in range(A):
                    pa = probs_or_pol[i, p, a]
                    if pa > 0.0:
                        j = nxt[i, p, a]
                        if j >= 0:
                            total += pa * ((1.0 if count_pieces else rew[i, p, a]) + V[j])
        out[i] = 0.5 * total
        d = abs(out[i] - V[i])
        if d > delta:
            delta = d
    return delta


def _evaluate(nxt, rew, policy, count_pieces, tol, max_sweeps):
    if policy.ndim == 2:                # deterministic: placement per (board, piece)
        arr = policy.astype(np.float64)[:, :, None]
        det = True
    else:                               # stochastic: probabilities (N, 2, A)
        arr = policy.astype(np.float64)
        det = False
    N = nxt.shape[0]
    V = np.zeros(N)
    new = np.zeros(N)
    for _ in range(max_sweeps):
        d = _eval_sweep(V, nxt, rew, arr, det, new, count_pieces)
        V, new = new, V
        if d < tol:
            break
    return V


def evaluate_policy(nxt, rew, policy, tol=1e-10, max_sweeps=100_000):
    """Iterative policy evaluation (RLI 4.1) on the afterstates.

    V(b) = 1/2 sum_p sum_u pi(u|b,p) [ r + V(b') ]. policy is either an
    array (N, 2) of placements or an array (N, 2, A) of probabilities.
    Returns V; V[0] is the expected number of rows from the empty board.
    """
    return _evaluate(nxt, rew, policy, False, tol, max_sweeps)


def expected_pieces(nxt, rew, policy, tol=1e-10, max_sweeps=100_000):
    """Expected number of pieces placed before game over, same recursion
    with reward 1 per piece that does not lose."""
    return _evaluate(nxt, rew, policy, True, tol, max_sweeps)


@njit(cache=True)
def _occ_step(pi_t, nxt, pol, out):
    N = nxt.shape[0]
    out[:] = 0.0
    for i in range(N):
        m = pi_t[i]
        if m > 0.0:
            for p in range(2):
                j = nxt[i, p, pol[i, p]]
                if j >= 0:
                    out[j] += 0.5 * m


def occupancy(nxt, pol, tol=1e-12, max_steps=1_000_000):
    """Expected number of visits to each board, starting from the empty one.

    Distribution propagation of the closed-loop Markov chain (deck 06):
    pi_{t+1}(j) = sum_i pi_t(i) P(i -> j), summed over t. The chain is
    absorbing (game over), so the mass left on the boards goes to zero.
    """
    N = nxt.shape[0]
    pi_t = np.zeros(N)
    pi_t[0] = 1.0
    visits = pi_t.copy()
    new = np.zeros(N)
    pol = pol.astype(np.int64)
    for _ in range(max_steps):
        _occ_step(pi_t, nxt, pol, new)
        pi_t, new = new, pi_t
        visits += pi_t
        if pi_t.sum() < tol:
            break
    return visits


@njit(cache=True)
def tie_random_probs(score, nxt):
    """Uniform probabilities over the placements with the highest score.

    Used for the baselines: random (score 0 everywhere) and greedy
    (score = rows cleared now, with losing placements put last).
    """
    N, _, A = nxt.shape
    probs = np.zeros((N, 2, A))
    for i in range(N):
        for p in range(2):
            best = -1e300
            for a in range(A):
                if score[i, p, a] > best:
                    best = score[i, p, a]
            n = 0
            for a in range(A):
                if score[i, p, a] == best:
                    n += 1
            for a in range(A):
                if score[i, p, a] == best:
                    probs[i, p, a] = 1.0 / n
    return probs


def baseline_probs(nxt, rew, name):
    """Probabilities of the two baselines on the table.

    random: every placement with probability 1/A.
    greedy: the greedy policy of deck 09, maximise the immediate reward
            (rows cleared now) and nothing else; ties at random; a
            placement that ends the game counts as worse than any other.
    """
    if name == 'random':
        score = np.zeros(nxt.shape)
    elif name == 'greedy':
        score = rew.astype(np.float64) - 1000.0 * (nxt < 0)
    else:
        raise ValueError(name)
    return tie_random_probs(score, nxt)


def agreement(Wstar, nxt, rew, pol, weights, tol=1e-9):
    """Fraction of (board, piece), weighted by weights[board], where pol
    picks a placement that is optimal (within tol of the best r + W*)."""
    N, _, A = nxt.shape
    q = rew + np.where(nxt >= 0, Wstar[np.maximum(nxt, 0)], 0.0)
    q = np.where(nxt >= 0, q, 0.0)
    best = q.max(axis=2)
    chosen = np.take_along_axis(q, pol.astype(np.int64)[:, :, None], axis=2)[:, :, 0]
    ok = (chosen >= best - tol).mean(axis=1)
    return float((weights * ok).sum() / weights.sum())


# ---------------------------------------------------------------- cache

def solve(W, H, verbose=False):
    """Table, optimal W* and optimal policy, cached in results/dp_cache."""
    path = CACHE / f'W{W}H{H}.npz'
    if path.exists():
        d = np.load(path)
        return dict(keys=d['keys'], nxt=d['nxt'], rew=d['rew'],
                    Wstar=d['Wstar'], deltas=d['deltas'], pol=d['pol'])
    keys, nxt, rew = build_table(W, H, verbose)
    t0 = time.perf_counter()
    Wstar, deltas = value_iteration(nxt, rew)
    if verbose:
        print(f'  value iteration: {len(deltas)} sweeps, '
              f'{time.perf_counter() - t0:.1f} s, W*(empty) = {Wstar[0]:.4f}')
    pol = greedy_from_W(Wstar, nxt, rew)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=keys, nxt=nxt, rew=rew, Wstar=Wstar, deltas=deltas, pol=pol)
    return dict(keys=keys, nxt=nxt, rew=rew, Wstar=Wstar, deltas=deltas, pol=pol)
