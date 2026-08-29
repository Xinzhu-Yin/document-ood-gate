# Development-only single-signal checkpoint

Date: 2026-08-12  
Scope: 306 N / 974 O development rows; 70% sealed test not scored.  
Model: frozen `microsoft/dit-base-finetuned-rvlcdip`.  
Threshold guardrail: 8,000 RVL validation pages, fixed at 99% acceptance.  
T guardrail: 10,000 balanced RVL test pages.

| Signal | Reference bank | AUROC | AUROC 95% CI | FPR95 | T false rejection |
|---|---:|---:|---:|---:|---:|
| MSP | - | 0.710 | 0.674--0.743 | 0.855 | 1.10% |
| Energy | - | 0.728 | 0.694--0.761 | 0.861 | 1.13% |
| Maximum logit | - | 0.720 | 0.687--0.753 | 0.845 | 1.10% |
| Negative entropy | - | 0.715 | 0.679--0.749 | 0.848 | 1.20% |
| Top-2 margin | - | 0.703 | 0.666--0.739 | 0.869 | 1.03% |
| Feature norm | - | 0.488 | 0.450--0.526 | 0.993 | 0.96% |
| kNN, k=10 | 8k | 0.724 | 0.687--0.759 | 0.875 | 0.90% |
| kNN, k=10 | 32k | 0.735 | 0.699--0.772 | 0.868 | 0.97% |
| kNN, k=10 | 80k | 0.743 | 0.707--0.777 | 0.851 | 0.89% |
| Mahalanobis | 8k | 0.742 | 0.707--0.775 | 0.830 | 1.01% |
| Mahalanobis | 32k | 0.745 | 0.711--0.778 | **0.816** | 0.89% |
| Mahalanobis | 80k | **0.746** | 0.714--0.779 | 0.831 | 0.92% |

AUROC/FPR95 above measure development N versus O. The T false-rejection rate uses a
separate threshold fixed on RVL validation, so it must not be interpreted as FPR95.
Bootstrap confidence intervals use 2,000 paired stratified resamples with seed 20260812.

The improvement is modest and the confidence intervals overlap. These results support
reviewing Mahalanobis and kNN, but do not justify a final claim until the next protocol is
frozen and the sealed test is evaluated exactly once.
