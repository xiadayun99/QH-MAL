from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_mec.config import MECConfig


EXPECTED_PACKAGES = {
    "numpy": "2.2.5",
    "pandas": "2.3.3",
    "scipy": "1.15.3",
    "matplotlib": "3.10.8",
    "seaborn": "0.13.2",
    "torch": "2.11.0",
}
EXPECTED_CONFIG = {
    "episodes": 600,
    "eval_episodes": 80,
    "episode_length": 80,
    "replay_buffer_size": 200000,
    "batch_size": 256,
    "user_actor_lr": 8e-05,
    "server_actor_lr": 4e-05,
    "critic_lr": 8e-05,
    "user_actor_update_interval": 1,
    "server_actor_update_interval": 2,
}
EXPECTED_MEC = {
    "num_users": 20,
    "num_servers": 4,
    "slot_duration_s": 1.0,
    "server_cpu_hz": 12.0e9,
    "queue_capacity_cycles": 1.2e10,
    "price_min": 0.1,
    "price_max": 1.0,
    "queue_penalty": 0.2,
    "fairness_penalty": 0.2,
    "violation_penalty": 1.2,
    "baseline_price_grid_size": 9,
    "baseline_best_response_rounds": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the paper environment and authoritative training preset.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on a missing or mismatched package version.")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info[:2] != (3, 10):
        failures.append(f"expected Python 3.10, found {platform.python_version()}")

    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        marker = "OK" if actual == expected else "CHECK"
        print(f"{package}: {actual} (expected {expected}) [{marker}]")
        if actual != expected:
            failures.append(f"{package}: expected {expected}, found {actual}")

    config_path = ROOT / "configs" / "paper_results.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            failures.append(f"paper_results.json {key}: expected {expected!r}, found {actual!r}")
    print(f"Paper config: {config_path} [{'OK' if not any('paper_results' in item for item in failures) else 'CHECK'}]")

    mec = MECConfig()
    for key, expected in EXPECTED_MEC.items():
        actual = getattr(mec, key)
        if actual != expected:
            failures.append(f"MECConfig {key}: expected {expected!r}, found {actual!r}")
    print(f"MEC defaults: [{'OK' if not any('MECConfig' in item for item in failures) else 'CHECK'}]")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    if failures:
        print("\nDifferences:")
        for failure in failures:
            print(f"- {failure}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nEnvironment and paper preset match the reference setup.")


if __name__ == "__main__":
    main()
