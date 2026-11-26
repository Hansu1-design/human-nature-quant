# human-nature-quant

> **"You cannot predict a person; you can only measure an era."**

A falsifiable framework — **Human Nature Theory** — about how the *frequency
distribution of human buying/selling behavior drifts over time under given
objective constraints*, applied to China A-share daily data.

The full theoretical skeleton (axioms, coordinate system, gates, falsification
criteria, anti-overfitting design) lives in [README.md](README.md) (Chinese,
~2500 lines). This file is the English orientation.

## What it is — and what it is not

| It is | It is not |
|---|---|
| A set of falsifiable statements about *population-level* behavior distributions | A prediction model for individuals |
| A statistical posture: define macroscopic quantities via distributions | Thermodynamics (markets have no conserved quantities, no equilibrium) |
| Measurement of the *shape* of crowd positioning | "Smart money behavioralism" (actors are unidentifiable from daily data) |

## Core commitments

- **Axiom L0**: individuals are unmeasurable from daily/minute bars. Every
  "the main force is washing/shipping" style assertion is therefore out of
  scope by construction.
- **One-shot falsifiability**: every claim in the skeleton carries an
  observable that, if seen, kills the theory. A statement that cannot be
  killed by one observation is data organization, not theory.
- **Pre-registered coordinate budget**: at daily frequency at most ~3
  independent dimensions are statistically identifiable; the 4th onward has
  standard errors exceeding estimates. New coordinates are a limited,
  pre-declared resource — not a free search.
- **Anti-fake-drift design**: three-layer defense (pre-registration,
  coordinate budget, out-of-sample era split) against the classic
  "re-fit until it works" failure mode.

## Repository layout

- `README.md` — the theory skeleton v0.1 (Chinese; axioms, gates, criteria)
- `d122_engine.py` — the decision engine implementing the gate/coordinate logic
- `build*.py` — data snapshot builders
- `d1xx_*.py` — numbered verification experiments (each answers one
  pre-registered question, results in matching `*_result.json`)
- `band_calib.py` — band calibration utilities

Data paths are hardcoded to a local research store (`G:/AStockData/...`);
repoint them to your own data source. Data files are deliberately not
shipped with this repository.

## Status

v0.1 — skeleton + engine + verification scripts. Entry-side gates validated;
exit-side layer under construction. See the skeleton's §13 pre-registration
for what counts as progress.

## License

MIT — see [LICENSE](LICENSE).
