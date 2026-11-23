"""Tabular reinforcement learning on the small boards (RLI ch. 5 and 6).

All agents interact with the game through the table built in dp.py, used
as a simulator: at each step they see the board index b and the piece p,
choose a placement a, and receive r = rew[b, p, a] and the next board
j = nxt[b, p, a] (j = -1: game over). The next piece is drawn at random.
They never use the probabilities or the exact values.

State S = (b, p), action A = placement, reward R = rows cleared, gamma = 1
(episodic task: every episode ends, see dp.safe_set).

Exploration: epsilon-greedy (RLI 2.2) with ties broken at random, and the
initial value q0 of the table (optimistic initial values, RLI 2.6).

  mc_control       on-policy Monte Carlo control, epsilon-soft (RLI 5.4),
                   every-visit, Q = average of the returns
  sarsa            Q(S,A) += alpha [R + Q(S',A') - Q(S,A)]        (RLI 6.4)
  q_learning       Q(S,A) += alpha [R + max_a Q(S',a) - Q(S,A)]   (RLI 6.5)
  expected_sarsa   Q(S,A) += alpha [R + sum_a pi(a|S') Q(S',a) - Q(S,A)]
                                                                  (RLI 6.6)
  td_afterstate    one value per board, the afterstate value W(b) (RLI 6.8):
                   when the piece p is drawn on board b,
                   W(b) += alpha [ max_a (r + W(b'_a)) - W(b) ]
                   i.e. a sample of the afterstate Bellman equation of dp.py,
                   with the expectation over p replaced by the piece drawn.
"""
import time

import numpy as np

from tetris import njit
import dp


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def argmax_random(q):
    """Index of the maximum of q, ties broken uniformly at random."""
    best = q[0]
    n = 1
    for a in range(1, q.shape[0]):
        if q[a] > best:
            best = q[a]
            n = 1
        elif q[a] == best:
            n += 1
    k = np.random.randint(n)
    for a in range(q.shape[0]):
        if q[a] == best:
            if k == 0:
                return a
            k -= 1
    return 0


@njit(cache=True)
def eps_greedy(q, eps):
    if np.random.random() < eps:
        return np.random.randint(q.shape[0])
    return argmax_random(q)


# ------------------------------------------------------------- Monte Carlo

@njit(cache=True)
def mc_control(Q, C, nxt, rew, episodes, eps, max_len=100_000):
    """On-policy every-visit MC control with epsilon-greedy policies.

    Generate an episode with the epsilon-greedy policy of Q; walking it
    backwards, G <- G + R_{t+1} and Q(S_t,A_t) <- Q + (G - Q) / C(S_t,A_t)
    (incremental average of the returns, RLI 2.4 and 5.4).
    """
    eb = np.zeros(max_len, dtype=np.int64)
    ep = np.zeros(max_len, dtype=np.int64)
    ea = np.zeros(max_len, dtype=np.int64)
    er = np.zeros(max_len)
    steps = 0
    for _ in range(episodes):
        b = 0
        T = 0
        while T < max_len:
            p = np.random.randint(2)
            a = eps_greedy(Q[b, p], eps)
            eb[T] = b
            ep[T] = p
            ea[T] = a
            er[T] = rew[b, p, a]
            T += 1
            j = nxt[b, p, a]
            if j < 0:
                break
            b = j
        steps += T
        G = 0.0
        for t in range(T - 1, -1, -1):
            G += er[t]
            b, p, a = eb[t], ep[t], ea[t]
            C[b, p, a] += 1
            Q[b, p, a] += (G - Q[b, p, a]) / C[b, p, a]
    return steps


# ---------------------------------------------------- one-step TD control

@njit(cache=True)
def sarsa(Q, nxt, rew, episodes, alpha, eps):
    steps = 0
    for _ in range(episodes):
        b = 0
        p = np.random.randint(2)
        a = eps_greedy(Q[b, p], eps)
        while True:
            steps += 1
            r = rew[b, p, a]
            j = nxt[b, p, a]
            if j < 0:                                   # terminal: Q(S', .) = 0
                Q[b, p, a] += alpha * (r - Q[b, p, a])
                break
            p2 = np.random.randint(2)
            a2 = eps_greedy(Q[j, p2], eps)
            Q[b, p, a] += alpha * (r + Q[j, p2, a2] - Q[b, p, a])
            b, p, a = j, p2, a2
    return steps


@njit(cache=True)
def q_learning(Q, nxt, rew, episodes, alpha, eps):
    steps = 0
    for _ in range(episodes):
        b = 0
        p = np.random.randint(2)
        while True:
            steps += 1
            a = eps_greedy(Q[b, p], eps)
            r = rew[b, p, a]
            j = nxt[b, p, a]
            if j < 0:                                   # terminal: Q(S', .) = 0
                Q[b, p, a] += alpha * (r - Q[b, p, a])
                break
            p2 = np.random.randint(2)                   # S' = (j, p2)
            Q[b, p, a] += alpha * (r + Q[j, p2].max() - Q[b, p, a])
            b, p = j, p2
    return steps


@njit(cache=True)
def expected_sarsa(Q, nxt, rew, episodes, alpha, eps):
    A = Q.shape[2]
    steps = 0
    for _ in range(episodes):
        b = 0
        p = np.random.randint(2)
        while True:
            steps += 1
            a = eps_greedy(Q[b, p], eps)
            r = rew[b, p, a]
            j = nxt[b, p, a]
            if j < 0:
                Q[b, p, a] += alpha * (r - Q[b, p, a])
                break
            p2 = np.random.randint(2)
            # expected value of Q(S', .) under the epsilon-greedy policy
            q2 = Q[j, p2]
            expected = (1.0 - eps) * q2.max() + eps * q2.mean()
            Q[b, p, a] += alpha * (r + expected - Q[b, p, a])
            b, p = j, p2
    return steps


# ------------------------------------------------------ TD(0) on afterstates

@njit(cache=True)
def td_afterstate(Wv, nxt, rew, episodes, alpha, eps):
    A = nxt.shape[2]
    q = np.zeros(A)
    steps = 0
    for _ in range(episodes):
        b = 0                                           # the empty board
        while True:
            steps += 1
            p = np.random.randint(2)                    # the next piece is drawn
            for a in range(A):
                j = nxt[b, p, a]
                q[a] = rew[b, p, a] + Wv[j] if j >= 0 else 0.0
            Wv[b] += alpha * (q.max() - Wv[b])          # sample of the Bellman eq.
            a = eps_greedy(q, eps)                      # the move actually played
            j = nxt[b, p, a]
            if j < 0:
                break
            b = j
    return steps


# ------------------------------------------------------------- training

METHODS = ('mc', 'sarsa', 'q_learning', 'expected_sarsa', 'td_afterstate')
LABELS = {'mc': 'Monte Carlo control', 'sarsa': 'Sarsa',
          'q_learning': 'Q-learning', 'expected_sarsa': 'Expected Sarsa',
          'td_afterstate': 'TD(0) on afterstates'}


@njit(cache=True)
def greedy_from_Q(Q):
    N, _, A = Q.shape
    pol = np.zeros((N, 2), dtype=np.int8)
    for i in range(N):
        for p in range(2):
            pol[i, p] = np.argmax(Q[i, p])
    return pol


def checkpoints(episodes, n=40, first=100):
    return np.unique(np.geomspace(first, episodes, n).astype(np.int64))


def train(method, table, episodes, alpha=0.1, eps=0.1, q0=0.0, seed_=0,
          n_checkpoints=40, occ_star=None):
    """Train one agent, evaluating its greedy policy exactly along the way.

    table: the dict of dp.solve. Returns (curve, final, learned), where
    curve has one row per checkpoint (episodes, steps, exact expected rows
    of the greedy policy, and for td_afterstate the error of W against W*
    weighted by occ_star, the visits of the optimal policy).
    """
    nxt, rew, Wstar = table['nxt'], table['rew'], table['Wstar']
    N, _, A = nxt.shape
    seed(seed_)
    if method == 'td_afterstate':
        Wv = np.full(N, float(q0))
    else:
        Q = np.full((N, 2, A), float(q0))
        if method == 'mc':
            C = np.zeros((N, 2, A), dtype=np.int64)
    curve = []
    done = 0
    steps = 0
    t_train = 0.0
    for ck in checkpoints(episodes, n_checkpoints):
        n = int(ck - done)
        t0 = time.perf_counter()
        if method == 'mc':
            steps += mc_control(Q, C, nxt, rew, n, eps)
        elif method == 'sarsa':
            steps += sarsa(Q, nxt, rew, n, alpha, eps)
        elif method == 'q_learning':
            steps += q_learning(Q, nxt, rew, n, alpha, eps)
        elif method == 'expected_sarsa':
            steps += expected_sarsa(Q, nxt, rew, n, alpha, eps)
        elif method == 'td_afterstate':
            steps += td_afterstate(Wv, nxt, rew, n, alpha, eps)
        else:
            raise ValueError(method)
        t_train += time.perf_counter() - t0
        done = int(ck)
        if method == 'td_afterstate':
            pol = dp.greedy_from_W(Wv, nxt, rew)
        else:
            pol = greedy_from_Q(Q)
        value = dp.evaluate_policy(nxt, rew, pol, tol=1e-8)[0]
        row = dict(episodes=done, steps=steps, value=value)
        if method == 'td_afterstate' and occ_star is not None:
            err = Wv - Wstar
            row['w_rmse'] = float(np.sqrt((occ_star * err ** 2).sum() / occ_star.sum()))
        curve.append(row)
    occ = dp.occupancy(nxt, pol)
    final = dict(method=method, value=curve[-1]['value'],
                 ratio=curve[-1]['value'] / Wstar[0],
                 pieces=float(dp.expected_pieces(nxt, rew, pol, tol=1e-8)[0]),
                 agreement=dp.agreement(Wstar, nxt, rew, pol, occ),
                 episodes=done, steps=steps, train_seconds=t_train)
    learned = dict(pol=pol)
    if method == 'td_afterstate':
        learned['W'] = Wv
    return curve, final, learned
