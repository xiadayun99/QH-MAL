# Manuscript Reference Values

These compact tables transcribe the values reported in the final manuscript. They are comparison targets, not cached outputs consumed by the training or plotting code.

- `overall_performance_reference.csv`: five-seed means and 95% CI half-widths.
- `queue_awareness_reference.csv`: paired QH-MAL versus QU-QH-MAL results.
- `stress_test_reference.csv`: seed-7 fixed-policy stress result.

`Server Return` denotes $\Pi_s^t$ in the manuscript. It is distinct from the monetary revenue term.

Small numerical differences across GPU models, CUDA libraries, and deterministic-kernel support are expected. A regenerated mean should be judged against the reported interval and the qualitative ranking, not by bitwise equality.
