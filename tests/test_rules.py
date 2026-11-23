"""Rules of the game: pieces, drop, clearing, game over, features."""
import numpy as np

from tetris import (placements, n_placements, drop, pack, unpack, features,
                    n_features, board_from_text, board_to_text, GAME_OVER)


def place(text, W, piece, a):
    rows = board_from_text(text, W)
    cells = placements(W)[0]
    c = drop(rows, len(rows), (1 << W) - 1, cells[piece, a])
    return c, board_to_text(rows, W)


def test_number_of_placements():
    for W in (4, 6, 10):
        cells, orient, column = placements(W)
        assert cells.shape == (2, n_placements(W), 4, 2)
        assert (orient == 0).sum() == W - 2 and (orient == 1).sum() == W - 1
        assert cells[..., 0].max() == W - 1 and cells[..., 0].min() == 0
    assert n_placements(6) == 9 and n_placements(10) == 17


def test_shapes():
    empty = '......\n......\n......'
    # S flat at column 0, Z flat at column 0
    assert place(empty, 6, 0, 0) == (0, '......\n.##...\n##....')
    assert place(empty, 6, 1, 0) == (0, '......\n##....\n.##...')
    # S upright at column 0 (placement W-2 = 4), Z upright
    assert place(empty, 6, 0, 4) == (0, '#.....\n##....\n.#....')
    assert place(empty, 6, 1, 4) == (0, '.#....\n##....\n#.....')


def test_piece_rests_on_the_highest_contact():
    board = '......\n......\n......\n..#...'
    # S flat at column 1 covers columns 1, 2 (bottom) and 2, 3 (top):
    # its bottom cell in column 2 lands on the block
    assert place(board, 6, 0, 1) == (0, '......\n..##..\n.##...\n..#...')


def test_clear_rows():
    # S upright at column 1 fills column 2 in rows 0-1 and column 1 in rows 1-2
    board = '......\n#..###\n##.###'
    c, out = place(board, 6, 0, 4 + 1)
    assert c == 2
    assert out == '......\n......\n.#....'


def test_game_over_before_clearing():
    # the S would complete the bottom row but its top sticks out: game over
    board = '#...##\n##.###'
    rows = board_from_text(board, 6)
    before = rows.copy()
    cells = placements(6)[0]
    c = drop(rows, 2, 63, cells[0, 4 + 1])   # S upright at column 1
    assert c == GAME_OVER
    assert (rows == before).all()            # the board is left unchanged


def test_pack_unpack():
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 64, 6)
    key = pack(rows, 6)
    out = np.zeros(6, dtype=np.int64)
    unpack(key, 6, 6, out)
    assert (out == rows).all()


def test_features():
    board = '......\n.#....\n.#.#..\n##.#.#'
    rows = board_from_text(board, 6)
    f = np.zeros(n_features(6))
    features(rows, 6, 4, f)
    heights = np.array([1, 3, 0, 2, 0, 1]) / 4
    assert np.allclose(f[:6], heights)
    assert np.allclose(f[6:11], np.abs(np.diff(heights)))
    assert np.isclose(f[11], 3 / 4)          # max height
    assert np.isclose(f[12], 0 / 4)          # holes
    rows2 = board_from_text('.#....\n......\n#.....', 6)
    features(rows2, 6, 3, f)
    assert np.isclose(f[12], 2 / 3)          # two holes under column 1
    assert f[13] == 1.0


# -------- cross-check against an independent implementation (the first
# feasibility script: bitboard in one integer, pure Python)

S_H = ((0, 0), (1, 0), (1, 1), (2, 1))
S_V = ((0, 1), (0, 2), (1, 0), (1, 1))
Z_H = ((1, 0), (2, 0), (0, 1), (1, 1))
Z_V = ((0, 0), (0, 1), (1, 1), (1, 2))


def ref_actions(W):
    out = []
    for piece in ((S_H, S_V), (Z_H, Z_V)):
        acts = []
        for cells in piece:
            width = 1 + max(dx for dx, _ in cells)
            for x in range(W - width + 1):
                acts.append(tuple((x + dx, dy) for dx, dy in cells))
        out.append(acts)
    return out


def ref_step(board, cells, W, H):
    full = (1 << W) - 1

    def fits(y):
        for cx, dy in cells:
            r = y + dy
            if r < H and (board >> (W * r + cx)) & 1:
                return False
        return True
    y = H
    while y > 0 and fits(y - 1):
        y -= 1
    for cx, dy in cells:
        if y + dy >= H:
            return None, 0
        board |= 1 << (W * (y + dy) + cx)
    rows = [(board >> (W * r)) & full for r in range(H)]
    kept = [row for row in rows if row != full]
    nb = 0
    for r, row in enumerate(kept):
        nb |= row << (W * r)
    return nb, H - len(kept)


def test_against_reference_implementation():
    rng = np.random.default_rng(1)
    W, H = 6, 6
    cells = placements(W)[0]
    ref = ref_actions(W)
    rows = np.zeros(H, dtype=np.int64)
    for _ in range(3000):
        key = int(rng.integers(0, 1 << (W * (H - 2))))   # random lower part
        p = int(rng.integers(2))
        for a in range(n_placements(W)):
            unpack(key, W, H, rows)
            c = drop(rows, H, (1 << W) - 1, cells[p, a])
            nb, rc = ref_step(key, ref[p][a], W, H)
            if nb is None:
                assert c == GAME_OVER
            else:
                assert c == rc and pack(rows, W) == nb
