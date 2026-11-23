"""The tabular methods of tabular.py on a board with too many boards for a
table built in advance (10x20): the table holds only the boards met.

Same five learners, same updates, same epsilon-greedy with random ties.
Two changes only:
  - the next board comes from playing the piece on the real board
    (tetris.drop), not from the table nxt/rew of dp.py;
  - the index of a board is found in a hash table (dict) keyed by the
    exact board; a board reached for the first time gets a new index, and
    its values start at q0. A board that is only looked at (the
    afterstates compared by TD(0) on afterstates) is not added: if it is
    not in the table its value is q0.

A table generalises nothing: the value of a board is learned only from
visits to that same board. So what matters is how often the boards met
come back; train() records the size of the table and, when testing the
greedy policy, the share of the boards met that are in the table.
"""
import time

import numpy as np
from numba import types
from numba.typed import Dict

from tetris import njit, drop, placements, GAME_OVER
from tabular import seed, argmax_random, eps_greedy

KEY = types.UniTuple(types.int64, 4)
LOSE = -1e-12         # a losing placement, when testing the greedy policy


@njit(cache=True)
def key_of(rows, W, H):
    """The exact board as 4 integers (63 // W rows of W bits in each)."""
    rpw = 63 // W
    k = np.zeros(4, dtype=np.int64)
    for r in range(H):
        k[r // rpw] |= rows[r] << (W * (r % rpw))
    return (k[0], k[1], k[2], k[3])


@njit(cache=True)
def lookup(d, rows, W, H):
    """Index of the board, -1 if it is not in the table."""
    k = key_of(rows, W, H)
    if k in d:
        return d[k]
    return -1


@njit(cache=True)
def index_of(d, rows, W, H):
    """Index of the board, added to the table if new."""
    k = key_of(rows, W, H)
    if k in d:
        return d[k]
    i = len(d)
    d[k] = i
    return i


# ------------------------------------------------------------- Monte Carlo

@njit(cache=True)
def mc_control(Q, C, d, W, H, cells, max_steps, eps, max_len, room):
    """As tabular.mc_control. Stops after max_steps pieces (whole episodes)
    or when fewer than `room` free places are left in Q. Returns (steps, episodes)."""
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    eb = np.zeros(max_len, dtype=np.int64)
    ep = np.zeros(max_len, dtype=np.int64)
    ea = np.zeros(max_len, dtype=np.int64)
    er = np.zeros(max_len)
    steps = 0
    episodes = 0
    while steps < max_steps and len(d) + room < Q.shape[0]:
        rows[:] = 0
        b = index_of(d, rows, W, H)
        T = 0
        while T < max_len:
            p = np.random.randint(2)
            a = eps_greedy(Q[b, p], eps)
            c = drop(rows, H, full, cells[p, a])
            eb[T] = b
            ep[T] = p
            ea[T] = a
            er[T] = 0.0 if c == GAME_OVER else c
            T += 1
            if c == GAME_OVER:
                break
            b = index_of(d, rows, W, H)
        steps += T
        episodes += 1
        G = 0.0
        for t in range(T - 1, -1, -1):
            G += er[t]
            b, p, a = eb[t], ep[t], ea[t]
            C[b, p, a] += 1
            Q[b, p, a] += (G - Q[b, p, a]) / C[b, p, a]
    return steps, episodes


# ---------------------------------------------------- one-step TD control

@njit(cache=True)
def sarsa(Q, d, W, H, cells, max_steps, alpha, eps, room):
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    steps = 0
    episodes = 0
    while steps < max_steps and len(d) + room < Q.shape[0]:
        rows[:] = 0
        b = index_of(d, rows, W, H)
        p = np.random.randint(2)
        a = eps_greedy(Q[b, p], eps)
        while True:
            steps += 1
            c = drop(rows, H, full, cells[p, a])
            if c == GAME_OVER:                          # terminal: Q(S', .) = 0
                Q[b, p, a] += alpha * (0.0 - Q[b, p, a])
                break
            j = index_of(d, rows, W, H)
            p2 = np.random.randint(2)
            a2 = eps_greedy(Q[j, p2], eps)
            Q[b, p, a] += alpha * (c + Q[j, p2, a2] - Q[b, p, a])
            b, p, a = j, p2, a2
        episodes += 1
    return steps, episodes


@njit(cache=True)
def q_learning(Q, d, W, H, cells, max_steps, alpha, eps, room):
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    steps = 0
    episodes = 0
    while steps < max_steps and len(d) + room < Q.shape[0]:
        rows[:] = 0
        b = index_of(d, rows, W, H)
        p = np.random.randint(2)
        while True:
            steps += 1
            a = eps_greedy(Q[b, p], eps)
            c = drop(rows, H, full, cells[p, a])
            if c == GAME_OVER:
                Q[b, p, a] += alpha * (0.0 - Q[b, p, a])
                break
            j = index_of(d, rows, W, H)
            p2 = np.random.randint(2)                   # S' = (j, p2)
            Q[b, p, a] += alpha * (c + Q[j, p2].max() - Q[b, p, a])
            b, p = j, p2
        episodes += 1
    return steps, episodes


@njit(cache=True)
def expected_sarsa(Q, d, W, H, cells, max_steps, alpha, eps, room):
    full = (1 << W) - 1
    rows = np.zeros(H, dtype=np.int64)
    steps = 0
    episodes = 0
    while steps < max_steps and len(d) + room < Q.shape[0]:
        rows[:] = 0
        b = index_of(d, rows, W, H)
        p = np.random.randint(2)
        while True:
            steps += 1
            a = eps_greedy(Q[b, p], eps)
            c = drop(rows, H, full, cells[p, a])
            if c == GAME_OVER:
                Q[b, p, a] += alpha * (0.0 - Q[b, p, a])
                break
            j = index_of(d, rows, W, H)
            p2 = np.random.randint(2)
            q2 = Q[j, p2]
            expected = (1.0 - eps) * q2.max() + eps * q2.mean()
            Q[b, p, a] += alpha * (c + expected - Q[b, p, a])
            b, p = j, p2
        episodes += 1
    return steps, episodes


# ------------------------------------------------------ TD(0) on afterstates

@njit(cache=True)
def td_afterstate(Wv, d, W, H, cells, max_steps, alpha, eps, q0, room):
    full = (1 << W) - 1
    A = cells.shape[1]
    rows = np.zeros(H, dtype=np.int64)
    tmp = np.zeros(H, dtype=np.int64)
    q = np.zeros(A)
    steps = 0
    episodes = 0
    while steps < max_steps and len(d) + room < Wv.shape[0]:
        rows[:] = 0
        b = index_of(d, rows, W, H)                     # the empty board
        while True:
            steps += 1
            p = np.random.randint(2)                    # the next piece is drawn
            for a in range(A):
                tmp[:] = rows
                c = drop(tmp, H, full, cells[p, a])
                if c == GAME_OVER:
                    q[a] = 0.0
                else:
                    j = lookup(d, tmp, W, H)
                    q[a] = c + (Wv[j] if j >= 0 else q0)
            Wv[b] += alpha * (q.max() - Wv[b])          # sample of the Bellman eq.
            a = eps_greedy(q, eps)                      # the move actually played
            c = drop(rows, H, full, cells[p, a])
            if c == GAME_OVER:
                break
            b = index_of(d, rows, W, H)
        episodes += 1
    return steps, episodes


# ------------------------------------------------- testing the greedy policy

@njit(cache=True)
def play_greedy(is_q, Q, Wv, d, W, H, cells, games, q0, max_pieces):
    """Play games with the greedy policy of the table (ties at random: on a
    board not in the table every placement ties). Q methods: argmax Q(b,p,.);
    TD(0) on afterstates: argmax r + W(b'), a losing placement last.
    Returns rows and pieces per game, the boards met that were in the
    table, and the boards met. (Q or Wv is unused, as is_q says.)"""
    full = (1 << W) - 1
    A = cells.shape[1]
    rows = np.zeros(H, dtype=np.int64)
    tmp = np.zeros(H, dtype=np.int64)
    q = np.zeros(A)
    out_rows = np.zeros(games, dtype=np.int64)
    out_pieces = np.zeros(games, dtype=np.int64)
    known = 0
    decisions = 0
    for g in range(games):
        rows[:] = 0
        total = 0
        n = 0
        while n < max_pieces:
            p = np.random.randint(2)
            b = lookup(d, rows, W, H)
            if b >= 0:
                known += 1
            decisions += 1
            for a in range(A):
                if is_q:                                # argmax_a Q(b, p, a)
                    q[a] = Q[b, p, a] if b >= 0 else q0
                    continue
                tmp[:] = rows                           # max_a r + W(b'_a)
                c = drop(tmp, H, full, cells[p, a])
                if c == GAME_OVER:
                    q[a] = LOSE
                else:
                    j = lookup(d, tmp, W, H)
                    q[a] = c + (Wv[j] if j >= 0 else q0)
            a = argmax_random(q)
            c = drop(rows, H, full, cells[p, a])
            if c == GAME_OVER:
                break
            total += c
            n += 1
        out_rows[g] = total
        out_pieces[g] = n
    return out_rows, out_pieces, known, decisions


# ---------------------------------------------------------------- training

def _grow(x, n, fill):
    out = np.full((n,) + x.shape[1:], fill, dtype=x.dtype)
    out[:x.shape[0]] = x
    return out


def evaluate(method, V, d, W, H, games, q0=0.0, seed_=0, max_pieces=10**6):
    cells = placements(W)[0]
    seed(seed_)
    is_q = method != 'td_afterstate'
    Q = V if is_q else np.zeros((1, 2, cells.shape[1]), dtype=np.float32)
    Wv = np.zeros(1, dtype=np.float32) if is_q else V
    r, n, known, dec = play_greedy(is_q, Q, Wv, d, W, H, cells, games, q0, max_pieces)
    s = r.std(ddof=1) if games > 1 else 0.0
    return dict(value=float(r.mean()), ci95=float(1.96 * s / np.sqrt(games)),
                pieces=float(n.mean()),
                coverage=float(known / max(1, dec)))


def train(method, W, H, steps, alpha=0.1, eps=0.1, q0=0.0, seed_=0,
          n_checkpoints=20, eval_games=200, final_games=2000, first=10_000,
          max_len=100_000, capacity=1_000_000):
    """Train for `steps` pieces; at each checkpoint test the greedy policy
    by Monte Carlo (eval_games games) and record the size of the table.
    Returns (curve, final, learned)."""
    cells = placements(W)[0]
    A = cells.shape[1]
    seed(seed_)
    d = Dict.empty(key_type=KEY, value_type=types.int64)
    if method == 'td_afterstate':
        V = np.full(capacity, q0, dtype=np.float32)
    else:
        V = np.full((capacity, 2, A), q0, dtype=np.float32)
        if method == 'mc':
            C = np.zeros((capacity, 2, A), dtype=np.int32)
    room = max_len + 1
    curve = []
    done = episodes = 0
    t_train = 0.0
    for k, ck in enumerate(np.unique(np.geomspace(first, steps, n_checkpoints).astype(np.int64))):
        while done < ck:
            if len(d) + room >= V.shape[0]:             # make room in the table
                n = int(V.shape[0] * 1.5) + room
                V = _grow(V, n, q0)
                if method == 'mc':
                    C = _grow(C, n, 0)
            t0 = time.perf_counter()
            todo = int(ck - done)
            if method == 'mc':
                s, e = mc_control(V, C, d, W, H, cells, todo, eps, max_len, room)
            elif method == 'sarsa':
                s, e = sarsa(V, d, W, H, cells, todo, alpha, eps, room)
            elif method == 'q_learning':
                s, e = q_learning(V, d, W, H, cells, todo, alpha, eps, room)
            elif method == 'expected_sarsa':
                s, e = expected_sarsa(V, d, W, H, cells, todo, alpha, eps, room)
            elif method == 'td_afterstate':
                s, e = td_afterstate(V, d, W, H, cells, todo, alpha, eps, q0, room)
            else:
                raise ValueError(method)
            t_train += time.perf_counter() - t0
            done += s
            episodes += e
        row = dict(steps=done, episodes=episodes, boards=len(d))
        row.update(evaluate(method, V, d, W, H, eval_games, q0, seed_=10_000 + k))
        # the test games reseed the random generator: training goes on from
        # a seed of its own, so that different seeds stay independent runs
        seed(1_000_003 * (seed_ + 1) + k)
        curve.append(row)
    final = dict(method=method, W=W, H=H, steps=done, episodes=episodes,
                 boards=len(d), train_seconds=t_train,
                 **evaluate(method, V, d, W, H, final_games, q0, seed_=999))
    learned = dict(V=V[:len(d)], d=d)
    return curve, final, learned
