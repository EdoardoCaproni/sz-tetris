"""Feasibility check: exact DP on SZ-Tetris with a reduced board.

Rules assumed: only S and Z pieces, drawn uniformly at random; the player
chooses orientation and column, the piece drops straight down; full rows are
cleared; the game ends when a piece does not fit entirely inside the board.
Objective: expected number of cleared rows until game over (undiscounted).
"""
import sys
import time

import numpy as np

W = int(sys.argv[1])
H = int(sys.argv[2])
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000

FULL = (1 << W) - 1

# cells as (dx, dy), dy grows upwards
S_H = ((0, 0), (1, 0), (1, 1), (2, 1))
S_V = ((0, 1), (0, 2), (1, 0), (1, 1))
Z_H = ((1, 0), (2, 0), (0, 1), (1, 1))
Z_V = ((0, 0), (0, 1), (1, 1), (1, 2))
PIECES = ((S_H, S_V), (Z_H, Z_V))


def placements(piece):
    out = []
    for cells in piece:
        width = 1 + max(dx for dx, _ in cells)
        for x in range(W - width + 1):
            out.append(tuple((x + dx, dy) for dx, dy in cells))
    return out


ACTIONS = [placements(p) for p in PIECES]
NA = max(len(a) for a in ACTIONS)


def fits(board, cells, y):
    for cx, dy in cells:
        r = y + dy
        if r < H and (board >> (W * r + cx)) & 1:
            return False
    return True


def step(board, cells):
    """Return (next_board, rows_cleared), or (None, 0) on game over."""
    y = H
    while y > 0 and fits(board, cells, y - 1):
        y -= 1
    for cx, dy in cells:
        r = y + dy
        if r >= H:
            return None, 0
        board |= 1 << (W * r + cx)
    rows = [(board >> (W * r)) & FULL for r in range(H)]
    kept = [row for row in rows if row != FULL]
    cleared = H - len(kept)
    nb = 0
    for r, row in enumerate(kept):
        nb |= row << (W * r)
    return nb, cleared


t0 = time.time()
index = {0: 0}
frontier = [0]
nxt_rows = []   # per board: list over pieces of list over actions (next index or -1)
rew_rows = []
boards = [0]
i = 0
aborted = False
while i < len(boards):
    b = boards[i]
    nrow = np.full((2, NA), -2, dtype=np.int64)  # -2 = action not available
    rrow = np.zeros((2, NA), dtype=np.int8)
    for p in range(2):
        for a, cells in enumerate(ACTIONS[p]):
            nb, c = step(b, cells)
            if nb is None:
                nrow[p, a] = -1
            else:
                j = index.get(nb)
                if j is None:
                    j = len(boards)
                    index[nb] = j
                    boards.append(nb)
                nrow[p, a] = j
                rrow[p, a] = c
    nxt_rows.append(nrow)
    rew_rows.append(rrow)
    i += 1
    if len(boards) > CAP:
        aborted = True
        break

print(f"W={W} H={H}: reachable boards = {len(boards)}"
      f"{' (ABORTED at cap, lower bound)' if aborted else ''}"
      f", states (board x piece) = {2 * len(boards)}, BFS {time.time() - t0:.1f}s")
if aborted:
    sys.exit(0)

N = len(boards)
nxt = np.stack(nxt_rows)          # (N, 2, NA)
rew = np.stack(rew_rows).astype(np.float64)
avail = nxt != -2
term = nxt == -1
idx = np.where(nxt >= 0, nxt, N)  # N = terminal slot with value 0

Wv = np.zeros(N + 1)
t0 = time.time()
for sweep in range(1, 2_000_001):
    q = rew + Wv[idx]
    q[term] = 0.0
    q[~avail] = -np.inf
    new = q.max(axis=2).mean(axis=1)
    delta = np.max(np.abs(new - Wv[:N]))
    Wv[:N] = new
    if delta < 1e-10:
        break
print(f"  value iteration: {sweep} sweeps, delta {delta:.2e}, {time.time() - t0:.1f}s")
print(f"  optimal expected rows from the empty board = {Wv[0]:.4f}")

# expected number of pieces placed under the optimal policy
q = rew + Wv[idx]
q[term] = 0.0
q[~avail] = -np.inf
best = q.argmax(axis=2)                                   # (N, 2)
bn = np.take_along_axis(idx, best[:, :, None], axis=2)[:, :, 0]
bt = np.take_along_axis(term, best[:, :, None], axis=2)[:, :, 0]
L = np.zeros(N + 1)
for _ in range(2_000_000):
    newL = (np.where(bt, 0.0, 1.0 + L[bn])).mean(axis=1)
    d = np.max(np.abs(newL - L[:N]))
    L[:N] = newL
    if d < 1e-10:
        break
print(f"  expected pieces placed before game over (optimal policy) = {L[0]:.2f}")
