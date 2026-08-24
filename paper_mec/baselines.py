from __future__ import annotations

import numpy as np

from paper_mec.env import MECEnvironment


DPP_JPO = "DPP Joint Pricing-Offloading"
STACKELBERG_JPO = "Stackelberg Joint Pricing-Offloading"
MODEL_BASED_BASELINES = [DPP_JPO, STACKELBERG_JPO]


def _full_equal_allocation_controls(env: MECEnvironment) -> np.ndarray:
    """Return full-utilization, equal-split controls for nonlearning baselines."""
    return np.column_stack(
        [
            np.ones(env.cfg.num_servers, dtype=np.float64),
            np.zeros(env.cfg.num_servers, dtype=np.float64),
        ]
    )


def _local_cost(env: MECEnvironment, user: int) -> float:
    cfg = env.cfg
    slot = env.current_slot
    delay = slot.cycles[user] / env.local_cpu_hz[user]
    energy = cfg.energy_coeff * (env.local_cpu_hz[user] ** 2) * slot.cycles[user]
    latency_tol = slot.latency_tol_s[user]
    risk = max(delay / max(latency_tol, 1.0e-6) - cfg.risk_trigger_ratio, 0.0) ** 2
    return float(
        cfg.alpha_delay * delay
        + cfg.beta_energy * energy
        + cfg.violation_penalty * risk
    )


def _offload_cost(
    env: MECEnvironment,
    user: int,
    server: int,
    counts: np.ndarray,
    same_slot_workload_ahead: float,
    prices: np.ndarray,
    queue_aware: bool,
) -> float:
    cfg = env.cfg
    slot = env.current_slot
    uplink = slot.data_mb[user] * 8.0 / slot.uplink_mbps[user, server]
    downlink = cfg.output_ratio * slot.data_mb[user] * 8.0 / slot.downlink_mbps[user, server]
    wait = (
        (env.queues[server] + same_slot_workload_ahead) / cfg.server_cpu_hz
        if queue_aware
        else 0.0
    )
    execution = slot.cycles[user] / (
        cfg.server_cpu_hz / max(counts[server] + 1.0, 1.0)
    )
    delay = uplink + wait + execution + downlink
    energy = cfg.tx_power_w * (uplink + downlink)
    payment = prices[server] * (slot.cycles[user] / 1.0e9)
    latency_tol = slot.latency_tol_s[user]
    risk = max(delay / max(latency_tol, 1.0e-6) - cfg.risk_trigger_ratio, 0.0) ** 2
    return float(
        cfg.alpha_delay * delay
        + cfg.beta_energy * energy
        + cfg.eta_payment * payment
        + cfg.violation_penalty * risk
    )


def _estimate_cost(
    env: MECEnvironment,
    user: int,
    server: int,
    counts: np.ndarray,
    planned_cycles: np.ndarray,
    prices: np.ndarray,
    queue_aware: bool,
) -> tuple[float, int]:
    local_cost = _local_cost(env, user)
    off_cost = _offload_cost(
        env,
        user,
        server,
        counts,
        float(planned_cycles[server]),
        prices,
        queue_aware,
    )
    return (local_cost, 0) if local_cost <= off_cost else (off_cost, server + 1)


def _rotating_user_order(env: MECEnvironment) -> np.ndarray:
    """Return a deterministic rotating order to avoid fixed user priority."""
    start = env.slot_index % max(env.cfg.num_users, 1)
    return np.roll(np.arange(env.cfg.num_users), -start)


def _best_response_actions(
    env: MECEnvironment,
    prices: np.ndarray,
    *,
    queue_aware: bool,
    drift_weight: float = 0.0,
    penalty_scale: float = 1.0,
) -> np.ndarray:
    """Compute sequential user best responses for a fixed price vector.

    ``drift_weight`` adds the normalized marginal queue pressure used by the
    DPP-JPO baseline.  A value of zero gives the Stackelberg followers' myopic
    delay-energy-payment best response.
    """
    counts = np.zeros(env.cfg.num_servers, dtype=np.float64)
    planned_cycles = np.zeros(env.cfg.num_servers, dtype=np.float64)
    actions = np.zeros(env.cfg.num_users, dtype=np.int64)
    queue_ratio = env.queues / env.cfg.queue_capacity_cycles
    for user in _rotating_user_order(env):
        best_score = penalty_scale * _local_cost(env, int(user))
        best_action = 0
        for server in np.where(env.candidate_mask[user] > 0.5)[0]:
            base_cost = _offload_cost(
                env,
                int(user),
                int(server),
                counts,
                float(planned_cycles[server]),
                prices,
                queue_aware,
            )
            workload_slots = env.current_slot.cycles[user] / (
                env.cfg.server_cpu_hz * env.cfg.slot_duration_s
            )
            score = (
                penalty_scale * base_cost
                + drift_weight * queue_ratio[server] * workload_slots
            )
            if score < best_score - 1.0e-12:
                best_score = score
                best_action = int(server + 1)
        actions[user] = best_action
        if best_action > 0:
            counts[best_action - 1] += 1.0
            planned_cycles[best_action - 1] += env.current_slot.cycles[user]
    return actions


def _project_planned_actions(
    env: MECEnvironment,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply admission and return the common same-slot workload-ahead trace."""
    projected, _, _, _, workload_ahead = env._apply_admission_projection(actions)
    return projected, workload_ahead


def _predicted_profile(
    env: MECEnvironment,
    actions: np.ndarray,
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate the exact implemented one-slot queue update and costs.

    This is deliberately model based: the DPP/Stackelberg comparators receive
    the observed queue and apply the same known recurrence as the environment,
    rather than fitting a learned queue predictor.
    """
    projected, same_slot_workload_ahead = _project_planned_actions(env, actions)
    arrivals = np.zeros(env.cfg.num_servers, dtype=np.float64)
    revenue = np.zeros(env.cfg.num_servers, dtype=np.float64)
    user_costs = np.array(
        [_local_cost(env, user) for user in range(env.cfg.num_users)],
        dtype=np.float64,
    )
    counts = np.array(
        [np.count_nonzero(projected == server + 1) for server in range(env.cfg.num_servers)],
        dtype=np.float64,
    )
    for user, action in enumerate(projected):
        if action == 0:
            continue
        server = int(action - 1)
        arrivals[server] += env.current_slot.cycles[user]
        revenue[server] += prices[server] * (env.current_slot.cycles[user] / 1.0e9)
        user_costs[user] = _offload_cost(
            env,
            user,
            server,
            np.maximum(counts - 1.0, 0.0),
            float(same_slot_workload_ahead[user]),
            prices,
            queue_aware=True,
        )
    allocated_cpu_rates = np.zeros(env.cfg.num_servers, dtype=np.float64)
    for server in range(env.cfg.num_servers):
        assigned = np.where(projected == (server + 1))[0]
        if assigned.size > 0:
            shares = env._allocation_from_controls(
                assigned,
                server,
                utilization=1.0,
                temperature=0.0,
            )
            allocated_cpu_rates[server] = float(shares.sum())
        elif env.queues[server] > 0.0:
            # Nonlearning baselines use full aggregate capacity; carried-over
            # backlog is an active workload even without a new admission.
            allocated_cpu_rates[server] = env.cfg.server_cpu_hz
    _, next_queues = env._queue_transition(
        env.queues,
        arrivals,
        allocated_cpu_rates,
    )
    next_queues = np.minimum(next_queues, env.cfg.queue_capacity_cycles)
    return arrivals, revenue, next_queues, float(user_costs.mean())


def _price_grid(env: MECEnvironment) -> np.ndarray:
    return np.linspace(
        env.cfg.price_min,
        env.cfg.price_max,
        max(int(env.cfg.baseline_price_grid_size), 2),
        dtype=np.float64,
    )


def dpp_joint_pricing_offloading_policy(
    env: MECEnvironment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-aligned drift-plus-penalty joint pricing/offloading baseline.

    The controller minimizes a one-slot quadratic queue-drift surrogate plus
    ``V`` times mean user cost minus a weighted server-revenue term.  Prices
    are selected by deterministic coordinate search and users are assigned by
    marginal DPP cost.  It observes the actual current queue and evaluates the
    known next-queue equation exactly, making it a stronger model-based
    alternative than a controller supplied only with a queue estimate.  This
    is an executable adaptation to the paper's model, not a claim of
    reproducing every detail of any cited external system.
    """
    cfg = env.cfg
    load_ratio = np.clip(env._predicted_load() / cfg.server_cpu_hz, 0.0, 1.0)
    queue_ratio = np.clip(env.queues / cfg.queue_capacity_cycles, 0.0, 1.0)
    normalized = np.clip(
        cfg.price_load_weight * load_ratio + cfg.price_queue_weight * queue_ratio,
        0.0,
        1.0,
    )
    prices = cfg.price_min + (cfg.price_max - cfg.price_min) * normalized
    grid = _price_grid(env)

    def objective(candidate_prices: np.ndarray) -> tuple[float, np.ndarray]:
        actions = _best_response_actions(
            env,
            candidate_prices,
            queue_aware=True,
            drift_weight=cfg.dpp_queue_weight,
            penalty_scale=cfg.dpp_v,
        )
        _, revenue, next_queues, mean_user_cost = _predicted_profile(
            env, actions, candidate_prices
        )
        q0 = env.queues / cfg.queue_capacity_cycles
        q1 = next_queues / cfg.queue_capacity_cycles
        normalized_drift = 0.5 * float(np.square(q1).sum() - np.square(q0).sum())
        penalty = mean_user_cost - cfg.dpp_profit_weight * float(revenue.mean())
        return normalized_drift + cfg.dpp_v * penalty, actions

    for _ in range(max(int(cfg.baseline_best_response_rounds), 1)):
        for server in range(cfg.num_servers):
            best_price = float(prices[server])
            best_objective = float("inf")
            for candidate in grid:
                proposal = prices.copy()
                proposal[server] = candidate
                value, _ = objective(proposal)
                if value < best_objective - 1.0e-12:
                    best_objective = value
                    best_price = float(candidate)
            prices[server] = best_price

    _, actions = objective(prices)
    allocation_controls = _full_equal_allocation_controls(env)
    return actions, prices, allocation_controls


def stackelberg_joint_pricing_offloading_policy(
    env: MECEnvironment,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-aligned Stackelberg price-leader/user-follower baseline.

    Server leaders perform coordinate best responses on a public price grid.
    For every price proposal, users recompute queue-aware cost-minimizing
    offloading responses.  The server utility uses the same revenue and
    normalized post-decision queue penalty as the common environment.
    """
    cfg = env.cfg
    prices = env.prices.copy()
    grid = _price_grid(env)
    for _ in range(max(int(cfg.baseline_best_response_rounds), 1)):
        for server in range(cfg.num_servers):
            best_price = float(prices[server])
            best_utility = -float("inf")
            best_queue = float("inf")
            for candidate in grid:
                proposal = prices.copy()
                proposal[server] = candidate
                actions = _best_response_actions(
                    env,
                    proposal,
                    queue_aware=True,
                )
                _, revenue, next_queues, _ = _predicted_profile(env, actions, proposal)
                utility = (
                    revenue[server]
                    - cfg.queue_penalty
                    * next_queues[server]
                    / cfg.queue_capacity_cycles
                )
                next_queue = float(next_queues[server])
                if (
                    utility > best_utility + 1.0e-12
                    or (
                        abs(utility - best_utility) <= 1.0e-12
                        and next_queue < best_queue - 1.0e-9
                    )
                ):
                    best_utility = float(utility)
                    best_queue = next_queue
                    best_price = float(candidate)
            prices[server] = best_price

    actions = _best_response_actions(env, prices, queue_aware=True)
    allocation_controls = _full_equal_allocation_controls(env)
    return actions, prices, allocation_controls


def random_policy(env: MECEnvironment, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.full(env.cfg.num_servers, 0.55, dtype=np.float64)
    user_actions = np.zeros(env.cfg.num_users, dtype=np.int64)
    for user in range(env.cfg.num_users):
        valid_actions = [0] + [server + 1 for server in range(env.cfg.num_servers) if env.candidate_mask[user, server] > 0.5]
        user_actions[user] = int(rng.choice(valid_actions))
    allocation_controls = _full_equal_allocation_controls(env)
    return user_actions, prices, allocation_controls


def greedy_policy(
    env: MECEnvironment,
    queue_aware: bool,
    fixed_price: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.full(env.cfg.num_servers, fixed_price, dtype=np.float64)
    counts = np.zeros(env.cfg.num_servers, dtype=np.float64)
    planned_cycles = np.zeros(env.cfg.num_servers, dtype=np.float64)
    user_actions = np.zeros(env.cfg.num_users, dtype=np.int64)
    order = np.arange(env.cfg.num_users)
    env.rng.shuffle(order)
    for user in order:
        best_cost = float("inf")
        best_action = 0
        local_delay = env.current_slot.cycles[user] / env.local_cpu_hz[user]
        local_energy = env.cfg.energy_coeff * (env.local_cpu_hz[user] ** 2) * env.current_slot.cycles[user]
        latency_tol = env.current_slot.latency_tol_s[user]
        local_risk = max(local_delay / max(latency_tol, 1.0e-6) - env.cfg.risk_trigger_ratio, 0.0) ** 2
        local_cost = env.cfg.alpha_delay * local_delay + env.cfg.beta_energy * local_energy + env.cfg.violation_penalty * local_risk
        if local_cost < best_cost:
            best_cost = local_cost
            best_action = 0
        for server in np.where(env.candidate_mask[user] > 0.5)[0]:
            cost, action = _estimate_cost(
                env,
                user,
                server,
                counts,
                planned_cycles,
                prices,
                queue_aware,
            )
            if cost < best_cost:
                best_cost = cost
                best_action = action
        user_actions[user] = best_action
        if best_action > 0:
            counts[best_action - 1] += 1.0
            planned_cycles[best_action - 1] += env.current_slot.cycles[user]
    allocation_controls = _full_equal_allocation_controls(env)
    return user_actions, prices, allocation_controls
