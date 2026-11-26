# SZ-Tetris with Dynamic Programming and Reinforcement Learning

Project for the course *Dynamic Programming and Reinforcement Learning*
(University of Siena).

**▶ [Watch the policies play, and see every result, in the browser](https://edoardocaproni.github.io/sz-tetris/)**

SZ-Tetris is Tetris played with only the S and Z pieces. Szita and Szepesvári
(2010) proposed it as a benchmark for reinforcement learning: it keeps the
difficulty of Tetris and every game is doomed. An alternating sequence of S
and Z pieces always ends the game on boards of width 2n with n odd (Burgiel,
1997), so with random pieces the question is how long the best player
survives.

The project does three things:

1. **Exact solution.** On boards of width 6 and height 3 to 6 (up to 1.6
   million boards) the game is solved by value iteration.
2. **Reinforcement learning against the exact answer.** The tabular methods
   of the course learn the same boards by playing. Every learned policy is
   then scored **exactly** against the optimum, by policy evaluation on the
   table of all boards.
3. **Beyond tables.** Linear function approximation and REINFORCE work on
   board features, so they also run on 10×20, where no table is possible.

## Rules

- The next piece is S or Z with probability 1/2 each.
- The player chooses the orientation and the column; the piece drops straight
  down (no sliding under overhangs).
- Full rows are cleared and give one point each.
- The game ends when a piece does not fit entirely inside the board. This is
  checked before full rows are cleared: a piece that completes a row but
  sticks out above the top still ends the game.

## Formulation

| | |
|---|---|
| state | $x_t = (b_t, p_t)$: the board and the current piece |
| input | $u_t$: a placement of $p_t$, an orientation and a column ($2W-3$ choices) |
| dynamics | $b_{t+1} = f(b_t, p_t, u_t)$: drop and clear full rows |
| uncertainty | $p_{t+1} \in \lbrace S, Z\rbrace$, probability 1/2 each, independent of the past |
| reward | $r(b_t, p_t, u_t)$ = rows cleared |
| termination | game over: absorbing, no reward |
| objective | $\max E\left[\sum_t r_t\right]$ until game over, no discount |

**Afterstates.** The next piece is drawn after the move. So the value can be
defined on the board alone, before that piece is known: the afterstate, or
post-decision state. Its value is

$$W(b) = \frac{1}{2}\sum_{p\in\lbrace S,Z\rbrace}\max_{u}\Big[r(b,p,u) + W\big(f(b,p,u)\big)\Big], \qquad W(\text{game over}) = 0,$$

and the value of a state is $V(b,p) = \max_u [r(b,p,u) + W(f(b,p,u))]$. This
gives one entry per board instead of one per (board, piece). The expectation
is over the next piece only.

**Why value iteration converges without a discount.** The total-reward
setting needs every policy to reach game over with probability 1. The code
does not rely on Burgiel's theorem; it checks this directly. It computes the
largest set of boards from which some placement always avoids game over and
stays in the set, whatever piece comes ([`dp.safe_set`](src/dp.py)). The set
is empty on every board solved. So from every board some finite sequence of
pieces ends the game whatever the player does, and that sequence has
positive probability.

## Methods

| Method | Course topic | Code |
|---|---|---|
| Value iteration on afterstates, termination check | infinite horizon, total cost; RLI 4.4, 6.8 | [`dp.py`](src/dp.py) |
| Exact policy evaluation, expected visits (absorbing chain) | RLI 4.1; Markov chains | [`dp.py`](src/dp.py) |
| Greedy (most rows now) and random baselines | stochastic DP, greedy policy | [`dp.py`](src/dp.py), [`simulate.py`](src/simulate.py) |
| Heuristic: one-step look-ahead with a hand-made $\tilde J$ | approximation in value space | [`tetris.py`](src/tetris.py) |
| Monte Carlo estimates with 95% confidence intervals | simulation tools | [`simulate.py`](src/simulate.py) |
| On-policy Monte Carlo control (ε-soft) | RLI 5.4 | [`tabular.py`](src/tabular.py) |
| Sarsa, Q-learning, Expected Sarsa (ε-greedy, optimistic initial values) | RLI 2.6, 6.4–6.6 | [`tabular.py`](src/tabular.py) |
| TD(0) on afterstates | RLI 6.1, 6.8 | [`tabular.py`](src/tabular.py) |
| The same five on 10×20, with a hash table of the boards met | RLI 9.1: why generalisation is needed | [`tabular_hash.py`](src/tabular_hash.py) |
| Linear semi-gradient TD(0) on afterstates | RLI 9.3–9.4 | [`linear.py`](src/linear.py) |
| REINFORCE with baseline, softmax-linear policy | RLI 13.3–13.4 | [`linear.py`](src/linear.py) |

The linear methods use the Bertsekas–Tsitsiklis features: column heights,
differences of neighbouring heights, maximum height, holes, and a constant.

Four of the policies use the same rule, one-step look-ahead
$u = \arg\max_u [r + \tilde J(b')]$, and differ only in $\tilde J$:

- greedy: $\tilde J = 0$;
- heuristic: $\tilde J(b) = C - (\text{holes} + \tfrac12\sum_c|h_c - h_{c+1}| + 0.1\sum_c h_c)$,
  weights fixed in advance, not tuned;
- linear TD: the same features, learned weights;
- value iteration: $\tilde J = W^\ast$, exact.
The cross-entropy method (Szita and Lőrincz, 2006), the usual strong baseline
on Tetris, and deep Q-networks are outside the course and are not
implemented.

## Results

Expected rows cleared from the empty board, mean over 5 seeds (% of the
optimum in brackets).
- **6×H:** exact.
- **10×20:** Monte Carlo, 2,000 games per seed.
- **Tabular methods:** 10⁷ episodes each, with the best setting of a grid on 6×5;
  on 10×20, 10⁷ pieces each with a hash table of the boards met.

| Policy | 6×4 | 6×5 | 6×6 | 10×20 |
|---|---:|---:|---:|---:|
| Optimal (value iteration) | **4.33** | **7.43** | **11.38** | – |
| Greedy: most rows now | 0.50 (12%) | 0.53 (7.1%) | 0.60 (5.3%) | 0.14 |
| Random placement | 0.03 (0.7%) | 0.04 (0.6%) | 0.06 (0.5%) | 0.01 |
| Heuristic: holes, steps, height | 2.08 (48%) | 3.12 (42%) | 4.14 (36%) | 10.82 ± 0.03 |
| Monte Carlo control | 4.15 (96%) | 6.93 (93%) | 8.72 (77%) | 1.81 |
| Sarsa | 4.10 (95%) | 6.48 (87%) | 9.52 (84%) | 2.33 |
| Q-learning | 4.01 (93%) | 6.59 (89%) | 10.04 (88%) | 2.40 |
| Expected Sarsa | 4.02 (93%) | 6.47 (87%) | 9.32 (82%) | 2.43 |
| TD(0) on afterstates | 4.18 (97%) | 7.26 (98%) | 10.88 (96%) | 7.32 |
| Linear TD(0) on afterstates | – | 4.97 (67%) | 6.17 (54%) | 89.7 ± 1.6 |
| REINFORCE with baseline | – | 6.63 (89%) | 9.43 (83%) | 42.5 ± 1.1 |

On 10×20, ± is the 95% confidence interval of the estimate for one seed
(2,000 games). Greedy and random on 10×20 come from 100,000 games each.

What the numbers say:

- **Exact solution.** The optimum is 2.30, 4.33, 7.43 and 11.38 rows for
  heights 3 to 6. Value iteration converges in 9 to 169 sweeps, and the
  termination check is empty on every board. The greedy policy (best
  immediate reward) gets 5–12% of the optimum. The immediate reward is
  almost always 0, so a one-step look-ahead plays almost at random: the
  value of a board lies in the future.
- **Afterstates are what scales.** TD(0) on afterstates stays at 96–98%
  of the optimum on every board. The four methods that learn Q(b, p, u)
  drop from 93–96% on 6×4 to 77–88% on 6×6 with the same 10⁷ episodes.
  W has one entry per board, while Q has 2(2W − 3) = 18 per board.
  Also, each update of W uses the known effect of the placement instead
  of learning it.
- **No learner reaches 100%, even on 6×4.** Exploration never stops
  (constant ε). The TD methods also use a constant step size, which
  breaks the conditions Σα = ∞, Σα² < ∞, so their estimates keep moving.
  The learning curves on the page show the plateau.
- **Features cost a lot on small boards.** Linear TD with 14 features
  reaches only 67% / 54% of the optimum on 6×5 / 6×6. REINFORCE with the
  same features reaches 89% / 83%. A likely reason: acting greedily on
  approximate values can pick poor placements even when the values are
  roughly right, while the policy gradient improves the policy directly.
- **Tables do not carry over to 10×20.** With a hash table of the boards
  met, the tabular methods clear 1.8–2.4 rows (Q methods) and 7.3 (TD on
  afterstates) after 10⁷ pieces. Above greedy (0.14), below the heuristic
  (10.8). More than half of the pieces played in training (0.61–0.69)
  lead to a board never seen before, and the boards of the test games
  that are in the table (45–55%) are mostly the first, nearly empty ones.
  A table learns a board only from visits to that same board; features
  carry what is learned to similar boards.
- **10×20.** After 2.5·10⁸ pieces, linear TD clears 90 rows (seeds from
  87 to 92) and REINFORCE 42 (39 to 44); greedy 0.14, random 0.01. The
  two curves differ in shape. TD reaches about 90–100 rows within the
  first 10⁷ pieces and then moves between 70 and 105 with no trend: with
  a constant step size the weights never settle, and the greedy policy
  changes with them. REINFORCE climbs slowly and steadily and levels off
  near 42. TD learns from every piece; REINFORCE waits for the end of
  each game, and its returns get noisier as games get longer.
- **Step sizes.** Every setting is the best of a small grid, chosen by
  mean over seeds. For REINFORCE on 6×5 and 6×6 the largest step tried
  (0.1) is still the best, but the gain from 0.03 to 0.1 is only 0.7% and
  1.4% of the optimum. With step 0.01 on 10×20, 2 of 5 REINFORCE seeds
  collapsed after about 10⁸ pieces to a policy that clears no rows.
- **When to stop (10×20).** The budget is fixed in advance and then
  checked: for each seed, the mean of the last three checkpoints
  (1.2–2.5·10⁸ pieces) minus the mean of the three before (4.4–8.8·10⁷);
  a setting is still improving if the 95% interval of that difference
  over the seeds is above 0. REINFORCE at step 0.1 has levelled off
  (+0.6 ± 0.7 rows); smaller steps are still climbing from below and might
  end higher with a longer budget. Linear TD at step 0.001 comes out
  "still improving" (+4.7 ± 2.5), but its curve was already at 90–100
  rows after 10⁷ pieces: this is the drift described above, not
  learning. A first budget of 5·10⁷ pieces gave 88 and 36 rows.

## Repository

| Path | Content |
|---|---|
| [`src/tetris.py`](src/tetris.py) | rules: pieces, placements, drop, features |
| [`src/dp.py`](src/dp.py) | table of boards, value iteration, termination check, policy evaluation |
| [`src/tabular.py`](src/tabular.py) | MC control, Sarsa, Q-learning, Expected Sarsa, TD(0) on afterstates |
| [`src/linear.py`](src/linear.py) | linear TD(0), REINFORCE with baseline |
| [`src/simulate.py`](src/simulate.py) | playing games on any board, Monte Carlo estimates |
| [`experiments/`](experiments) | one script per experiment, results in `results/` |
| [`tests/`](tests) | rules (also against an independent implementation and the JavaScript engine), DP, learners |
| [`viewer/`](viewer) | the web page: the game in JavaScript and the results |

Setup and reproduction (Windows; Python 3.11):

```
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest tests
.venv\Scripts\python experiments\exp_dp.py          # exact solution, about 1 min
.venv\Scripts\python experiments\exp_tabular.py     # grid + final runs, about 25 min
.venv\Scripts\python experiments\exp_linear.py      # linear methods, about 3 h (10×20: 2 h on 20 processes)
.venv\Scripts\python experiments\export_viewer.py   # data of the page
```

## References

- I. Szita, C. Szepesvári, *SZ-Tetris as a benchmark for studying key problems
  of reinforcement learning*, ICML 2010 workshop on machine learning and games.
- H. Burgiel, *How to lose at Tetris*, The Mathematical Gazette 81 (1997).
- D. P. Bertsekas, J. N. Tsitsiklis, *Neuro-Dynamic Programming*, Athena
  Scientific (1996): the Tetris features.
- I. Szita, A. Lőrincz, *Learning Tetris using the noisy cross-entropy
  method*, Neural Computation 18 (2006).
- R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
- D. P. Bertsekas, *Reinforcement Learning and Optimal Control*.
