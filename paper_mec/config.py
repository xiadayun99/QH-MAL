from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MECConfig:
    num_users: int = 20
    num_servers: int = 4
    slot_duration_s: float = 1.0
    arrival_scale: float = 1.2
    # The training environment uses ``uniform`` arrivals and channel
    # goodputs.  The alternative distributions are reserved for explicit
    # out-of-distribution evaluation; they are never mixed into training by
    # the paper reproduction preset.
    arrival_distribution: str = "uniform"
    channel_distribution: str = "uniform"

    data_mb_min: float = 0.15
    data_mb_max: float = 0.60
    cycles_min: float = 0.8e9
    cycles_max: float = 2.2e9
    latency_min_s: float = 0.2
    latency_max_s: float = 1.0
    output_ratio: float = 0.1

    local_cpu_min_hz: float = 0.8e9
    local_cpu_max_hz: float = 1.2e9
    energy_coeff: float = 1.0e-28
    tx_power_w: float = 0.1

    # Bounds for observed effective goodput. The simulator resamples these
    # rates per slot instead of separately modeling PHY/MAC allocation.
    uplink_min_mbps: float = 10.0
    uplink_max_mbps: float = 28.0
    downlink_min_mbps: float = 14.0
    downlink_max_mbps: float = 36.0

    server_cpu_hz: float = 12.0e9
    # Physical workload-buffer capacity.  The default corresponds to one
    # second of service at the 12-GHz edge server.
    queue_capacity_cycles: float = 1.2e10

    price_min: float = 0.1
    price_max: float = 1.0
    queue_penalty: float = 0.2
    fairness_penalty: float = 0.2

    alpha_delay: float = 1.0
    beta_energy: float = 0.35
    eta_payment: float = 0.45
    violation_penalty: float = 1.2
    risk_trigger_ratio: float = 0.85

    price_load_weight: float = 0.38
    price_queue_weight: float = 0.42
    # The allocation actor controls both the aggregate CPU-utilization fraction
    # and the temperature used to distribute that aggregate rate across users.
    # A positive lower bound keeps execution-delay calculations finite for an
    # admitted task while still allowing the learned action to change service.
    cpu_utilization_min: float = 0.10
    alloc_temp_max: float = 8.0

    # Model-aligned online optimization baselines.  DPP-JPO and SG-JPO use
    # the same price grid and finite-buffer environment as every other method.
    baseline_price_grid_size: int = 9
    baseline_best_response_rounds: int = 2
    dpp_v: float = 1.0
    dpp_queue_weight: float = 2.0
    dpp_profit_weight: float = 0.25

    seed: int = 7


@dataclass
class TrainConfig:
    episodes: int = 220
    episode_length: int = 40
    eval_episodes: int = 30
    curve_eval_episodes: int = 2
    curve_eval_interval: int = 1
    curve_eval_arrival_scale: float = 1.2
    train_arrival_scales: tuple[float, ...] = (0.8, 1.0, 1.2, 1.4)
    selection_arrival_scales: tuple[float, ...] = (1.0, 1.2, 1.4)
    selection_eval_episodes: int = 4
    selection_interval: int = 10
    user_actor_hidden_dims: tuple[int, ...] = (192, 192, 128)
    server_actor_hidden_dims: tuple[int, ...] = (160, 160, 128)
    # Approximately parameter-matches the three role-specific actor networks
    # when the role-agnostic ablation replaces them with one shared network.
    role_agnostic_actor_hidden_dims: tuple[int, ...] = (288, 288, 224)
    critic_hidden_dims: tuple[int, ...] = (384, 384, 256)

    gamma: float = 0.99
    batch_size: int = 128
    replay_buffer_size: int = 4_000
    warmup_steps: int = 600
    updates_per_step: int = 1
    target_tau: float = 0.01
    # Queue-aware MATD3 baseline settings.  Target smoothing is applied only
    # to the continuous server controls; categorical user requests retain the
    # common Gumbel--Softmax relaxation used by every learning method.
    matd3_policy_delay: int = 2
    matd3_target_noise_std: float = 0.02
    matd3_target_noise_clip: float = 0.05

    user_actor_lr: float = 1.0e-4
    server_actor_lr: float = 5.0e-5
    critic_lr: float = 1.0e-4

    user_actor_update_interval: int = 1
    server_actor_update_interval: int = 2
    critic_update_interval: int = 1

    user_gumbel_tau: float = 0.55
    user_residual_scale: float = 0.10
    server_price_residual_scale: float = 0.03
    server_util_residual_scale: float = 0.15
    server_alloc_residual_scale: float = 0.015

    epsilon_start: float = 0.25
    epsilon_end: float = 0.03
    epsilon_decay_steps: int = 4_000
    gaussian_noise_std: float = 0.03

    critic_l2: float = 1.0e-4
    actor_l2: float = 5.0e-4
    grad_clip_norm: float = 1.0
    device: str = "auto"
    show_progress: bool = True
    progress_refresh_seconds: float = 5.0


@dataclass
class MethodSpec:
    name: str
    queue_visible: bool = True
    queue_in_delay: bool = True
    queue_penalty: float = 0.2
    fairness_penalty: float = 0.2
    two_timescale: bool = True
    # When true, a simultaneous-update ablation also sets the server actor
    # learning rate equal to the user actor learning rate.
    equal_actor_learning_rates: bool = False
    # False selects one role-conditioned, parameter-shared actor. Role-indexed
    # decoding still respects the categorical user and continuous domains.
    role_specific_actors: bool = True
    # qhmarl: proposed role-matched update schedule and twin critics
    # maddpg: single centralized critic per role and simultaneous actors
    # matd3: twin centralized critics, target smoothing, and common delay
    learner: str = "qhmarl"


@dataclass
class PathConfig:
    root: Path = Path(".")

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "artifacts" / "checkpoints"
