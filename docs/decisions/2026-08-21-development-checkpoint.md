# Development checkpoint decision — 2026-08-21

## Purpose

Record the project state after reviewing the first single-signal results. This is a
sequencing decision, not protocol v2, and it does not authorize a combination Gate or
sealed-test evaluation.

## Evidence reviewed

- The mandatory MSP/Energy reproduction gate passed green.
- Duplicate-aware splitting produced 1,294 development and 3,018 sealed rows.
- The primary development comparison excludes 14 rows from one contradictory exact-pixel
  N/O group, leaving 306 N and 974 O rows.
- Energy, kNN 80k, Mahalanobis 32k, and Mahalanobis 80k remain candidates; none has been
  selected as the final Gate.
- The 3,018-row sealed partition has not been scored.

## Decisions recorded

1. Repository presentation and engineering hygiene may be improved without changing
   experimental values, data membership, or scientific claims.
2. The shortcut/source audit is not run during this presentation-only step. Its later
   scheduling remains undecided, and protocol v1.1 still lists it as mandatory.
3. No combination Gate, input set, learner, hyperparameter, or selection rule is approved
   by this checkpoint.
4. The sealed test remains locked.

## Explicitly not decided

- Whether to run a learned combination Gate.
- Whether AUROC, FPR95, or a fixed operating point is the primary selection objective.
- How the N-95% benchmark threshold and RVL-validation-99% operating threshold will be
  presented in the next experiment.
- Which single-signal candidates advance to sealed evaluation.
- The conditions that authorize the one-time sealed evaluation.

## Next approval checkpoint

Before any combination experiment or sealed scoring, create and approve a separate frozen
protocol addendum that answers the five questions above. Until then, development artifacts
may be reviewed but no new candidate may use sealed-set feedback.

