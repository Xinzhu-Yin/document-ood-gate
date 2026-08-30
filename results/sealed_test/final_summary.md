# Final sealed-test results

- Freeze commit: `451fbdb4e902e75843e8e4ef5a31745cc95f1875`.
- Sealed extraction: exactly 3,018 images (694 N + 2,324 O), one forward each.
- Existing 32k reference cache reused; reference images re-extracted: 0.
- No post-sealed experimental iteration was performed.

## Core sealed baseline panel

| Method | AUROC [95% CI] | FPR95 [95% CI] |
|---|---|---|
| msp | 0.7372 [0.7160, 0.7605] | 0.8468 [0.7999, 0.8830] |
| energy | 0.7650 [0.7452, 0.7862] | 0.8369 [0.7388, 0.8873] |
| mahalanobis | 0.7474 [0.7253, 0.7697] | 0.8546 [0.8029, 0.9191] |
| knn | 0.7405 [0.7171, 0.7642] | 0.8795 [0.8352, 0.8993] |
| vim_origin_residual | 0.7433 [0.7213, 0.7658] | 0.8808 [0.8128, 0.9238] |
| vim | 0.7473 [0.7254, 0.7698] | 0.8731 [0.8068, 0.9204] |

## Replication predictions

- P1: **supported**
- P2: **supported**
- P3: **supported**
- P4: **supported**
- P5: **supported**

## Limitation

A full shortcut audit was not performed. The development/sealed split preserves source composition, reducing source-distribution mismatch across splits, but this does not rule out within-task acquisition, rendering, or source-related shortcuts. In particular, 395 recovered O pages are calibrated rerenders rather than byte-identical original TIFFs.
