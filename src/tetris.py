"""SZ-Tetris rules: pieces, placements, the drop, the features.

A board of width W and height H is an array of H row bitmasks (row 0 at
the bottom, bit c = column c). Rules (see the README):

- the current piece is S or Z; the player chooses an orientation and a
  column (a "placement") and the piece drops straight down;
- the game ends if, after the drop, a cell of the piece is above the top
  row. This is checked before full rows are cleared;
- otherwise full rows are cleared and give one point each.

In the professor's notation this file is the dynamics: given the board b
and the piece p, the input u (a placement) gives the next board
b' = f(b, p, u) and the reward r(b, p, u) = rows cleared. b' is the
afterstate: the board before the next piece is drawn.
"""
import os
from pathlib import Path

# numba cannot cache compiled code under a path with spaces or accents
# ("Progetti Università"), so the cache goes in the home folder
os.environ.setdefault('NUMBA_CACHE_DIR',
                      str(Path.home() / '.numba_cache' / 'sz-tetris'))

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

PIECES = 'SZ'
GAME_OVER = -1

# cells of each piece as (dx, dy), dy grows upwards
# S:   .XX      X.        Z:   XX.      .X
#      XX.      XX             .XX      XX
#               .X                      X.
SHAPES = (
    (((0, 0), (1, 0), (1, 1), (2, 1)),     # S flat
     ((0, 1), (0, 2), (1, 0), (1, 1))),    # S upright
    (((1, 0), (2, 0), (0, 1), (1, 1)),     # Z flat
     ((0, 0), (0, 1), (1, 1), (1, 2))),    # Z upright
)


def n_placements(W):
    """Placements per piece: W-2 columns flat, W-1 upright."""
    return (W - 2) + (W - 1)


def placements(W):
    """All placements on a board of width W.

    Returns cells[piece, a, k] = (x, dy) of the k-th cell, and for each
    placement a its orientation (0 flat, 1 upright) and leftmost column.
    The order is the same for S and Z: flat columns 0..W-3, then upright
    columns 0..W-2.
    """
    A = n_placements(W)
    cells = np.zeros((2, A, 4, 2), dtype=np.int64)
    orient = np.zeros(A, dtype=np.int64)
    column = np.zeros(A, dtype=np.int64)
    for p in range(2):
        a = 0
        for o in range(2):
            shape = SHAPES[p][o]
            width = 1 + max(dx for dx, _ in shape)
            for x in range(W - width + 1):
                for k, (dx, dy) in enumerate(shape):
                    cells[p, a, k] = (x + dx, dy)
                orient[a] = o
                column[a] = x
                a += 1
    return cells, orient, column


@njit(cache=True)
def _fits(rows, H, cells, y):
    for k in range(4):
        r = y + cells[k, 1]
        if r < H and (rows[r] >> cells[k, 0]) & 1:
            return False
    return True


@njit(cache=True)
def landing_row(rows, H, cells):
    """Row of the piece's bottom after dropping it straight down."""
    y = H
    while y > 0 and _fits(rows, H, cells, y - 1):
        y -= 1
    return y


@njit(cache=True)
def drop(rows, H, full, cells):
    """Drop a piece in place. Returns the rows cleared, or GAME_OVER.

    rows: the board (modified in place, unchanged on game over);
    full: the bitmask of a full row, 2**W - 1;
    cells: the (4, 2) cells of the placement.
    """
    y = landing_row(rows, H, cells)
    for k in range(4):
        if y + cells[k, 1] >= H:
            return GAME_OVER
    for k in range(4):
        rows[y + cells[k, 1]] |= 1 << cells[k, 0]
    kept = 0
    for r in range(H):
        if rows[r] != full:
            rows[kept] = rows[r]
            kept += 1
    for r in range(kept, H):
        rows[r] = 0
    return H - kept


@njit(cache=True)
def pack(rows, W):
    """Board -> integer key (row r in bits W*r .. W*r+W-1); needs W*H <= 63."""
    key = 0
    for r in range(rows.shape[0]):
        key |= np.int64(rows[r]) << (W * r)
    return key


@njit(cache=True)
def unpack(key, W, H, rows):
    mask = (1 << W) - 1
    for r in range(H):
        rows[r] = (key >> (W * r)) & mask


# ---------------------------------------------------------------- features
# Bertsekas and Tsitsiklis (1996) features for Tetris: column heights,
# absolute differences of neighbouring heights, maximum height, holes
# (empty cells below the top of their column), and a constant.
# Heights and holes are divided by H so that all features are O(1).

def n_features(W):
    return W + (W - 1) + 1 + 1 + 1


@njit(cache=True)
def features(rows, W, H, out):
    """Write the features of the board into out (length n_features(W))."""
    hmax = 0
    holes = 0
    for c in range(W):
        h = 0
        for r in range(H - 1, -1, -1):
            if (rows[r] >> c) & 1:
                h = r + 1
                break
        for r in range(h):
            if not (rows[r] >> c) & 1:
                holes += 1
        out[c] = h / H
        if h > hmax:
            hmax = h
    for c in range(W - 1):
        out[W + c] = abs(out[c] - out[c + 1])
    out[2 * W - 1] = hmax / H
    out[2 * W] = holes / H
    out[2 * W + 1] = 1.0


def heuristic_weights(W, H):
    """A hand-made cost-to-go on the same features, fixed in advance (not tuned):

      J~(b) = C - ( holes + 0.5 * sum_c |h_c - h_c+1| + 0.1 * sum_c h_c )   (in cells)

      holes      a hole cannot be filled until every row above it is cleared
      steps      S and Z land without a hole only on a step of the right
                 shape; big steps leave no such place
      heights    the distance to game over (a cleared row lowers the sum by W)
      C = 2WH    larger than any penalty, so every board is worth more than
                 game over (worth 0, as W(game over) = 0); the same for every
                 placement, so it changes no other choice

    Played as the one-step look-ahead max_u [ r + J~(b') ] (Bertsekas,
    approximation in value space), the same rule as linear TD with learned
    weights. The features are divided by H, so the weights are multiplied by H.
    """
    w = np.zeros(n_features(W))
    w[:W] = -0.1 * H                # column heights
    w[W:2 * W - 1] = -0.5 * H       # |height differences|
    w[2 * W] = -1.0 * H             # holes
    w[2 * W + 1] = 2.0 * W * H      # constant C
    return w


def board_to_text(rows, W):
    """Board as text, top row first (for debugging and tests)."""
    lines = []
    for r in range(len(rows) - 1, -1, -1):
        lines.append(''.join('#' if (int(rows[r]) >> c) & 1 else '.'
                             for c in range(W)))
    return '\n'.join(lines)


def board_from_text(text, W):
    """Inverse of board_to_text."""
    lines = [ln.strip() for ln in text.strip().split('\n')]
    rows = np.zeros(len(lines), dtype=np.int64)
    for i, ln in enumerate(reversed(lines)):
        assert len(ln) == W
        rows[i] = sum(1 << c for c, ch in enumerate(ln) if ch == '#')
    return rows


@njit(cache=True)
def dot(x, y):
    """x . y for 1-d arrays (numba's @ would need scipy)."""
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * y[i]
    return s
