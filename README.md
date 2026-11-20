# SZ-Tetris with Dynamic Programming and Reinforcement Learning

Project for the course *Dynamic Programming and Reinforcement Learning*
(University of Siena).

SZ-Tetris is Tetris played with only the S and Z pieces. It was proposed as a
benchmark for reinforcement learning by Szita and Szepesvári (2010). An
alternating sequence of S and Z pieces always ends the game on boards of width
2n with n odd (Burgiel, 1997), so with random pieces every game ends sooner or
later and the question is how long the best player can survive.

## Plan

1. **Exact solution.** On a narrow board (width 6, height up to 6) the game is
   small enough to be solved exactly with dynamic programming.
2. **Reinforcement learning.** The methods of the course are compared with the
   exact solution on the same boards.
3. **Larger boards.** The board grows until tabular methods stop working, and
   only function approximation is left.

## Rules used here

- The next piece is S or Z with probability 1/2 each.
- The player chooses the orientation and the column; the piece drops straight
  down.
- Full rows are cleared and give one point each.
- The game ends when a piece does not fit entirely inside the board.

## First numbers

`experiments/dp_feasibility.py` counts the states reachable from the empty
board and runs value iteration on them (width 6):

| Height | States    | Expected rows, optimal play | Pieces before game over |
|--------|-----------|-----------------------------|-------------------------|
| 3      | 660       | 2.30                        | 5.2                     |
| 4      | 12,698    | 4.33                        | 9.7                     |
| 5      | 183,316   | 7.43                        | 15.3                    |
| 6      | 3,155,370 | 11.38                       | 22.4                    |

These come from a first script without tests and still have to be confirmed.

## Setup

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python experiments\dp_feasibility.py 6 5
```

## References

- I. Szita, C. Szepesvári, *SZ-Tetris as a benchmark for studying key problems
  of reinforcement learning*, ICML 2010 workshop on machine learning and games.
- H. Burgiel, *How to lose at Tetris*, The Mathematical Gazette 81 (1997).
