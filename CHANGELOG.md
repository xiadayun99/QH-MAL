# Changelog

## 1.1.0 — 2026-08-23

- Aligned all public tables, figures, and reference files with the manuscript term `Server Return` for $\Pi_s^t$.
- Retained the internal `avg_profit` field so existing checkpoints and raw metric dictionaries remain compatible.
- Added legacy-column support so the final plotting pipeline can still read earlier CSV files containing `Profit`.
- Made the one-command runner pass the paper seeds, user counts, buffer windows, stress horizon, and load explicitly.
- Normalized Python source headers for clean cross-platform Git checkouts.
- Confirmed that the simulation model, algorithms, numerical settings, seeds, and reported reference values are unchanged from version 1.0.0.

## 1.0.0 — 2026-08-22

- Synced the public defaults to the final manuscript: nine methods and seeds `7, 11, 19, 23, 31`.
- Fixed the authoritative server actor learning rate at `4e-5`.
- Fixed the long-horizon stress protocol at `10,000` slots and arrival scale `1.6`.
- Removed the obsolete constrained comparator from the paper-facing method set.
- Added a one-command reproduction pipeline for main, population, buffer, stress, and ablation experiments.
- Replaced the legacy plotting entry with the exact final Fig. 2–7 pipeline.
- Added reference results, configuration checks, projection smoke testing, CI, and concise code–paper alignment notes.
