from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from paper_mec.config import MECConfig, MethodSpec


@dataclass
class SlotSample:
    data_mb: np.ndarray
    cycles: np.ndarray
    latency_tol_s: np.ndarray
    uplink_mbps: np.ndarray
    downlink_mbps: np.ndarray


class MECEnvironment:
    def __init__(self, config: MECConfig, rng: np.random.Generator | None = None) -> None:
        self.cfg = config
        self.rng = rng or np.random.default_rng(config.seed)
        self.local_cpu_hz = self.rng.uniform(
            config.local_cpu_min_hz,
            config.local_cpu_max_hz,
            size=config.num_users,
        )
        self.candidate_mask = np.zeros((config.num_users, config.num_servers), dtype=np.float64)
        coverage = min(2, config.num_servers)
        if config.num_servers == 4 and coverage == 2:
            for user in range(config.num_users):
                frac = user / max(config.num_users - 1, 1)
                if frac < 0.40:
                    start = 0
                elif frac < 0.80:
                    start = 1
                else:
                    start = 2
                self.candidate_mask[user, start : start + 2] = 1.0
        else:
            for user in range(config.num_users):
                offset = user % config.num_servers
                for step in range(coverage):
                    self.candidate_mask[user, (offset + step) % config.num_servers] = 1.0
        self.queues = np.zeros(config.num_servers, dtype=np.float64)
        self.prices = np.full(config.num_servers, 0.5 * (config.price_min + config.price_max))
        self.prev_load = np.zeros(config.num_servers, dtype=np.float64)
        self.prev_counts = np.zeros(config.num_servers, dtype=np.float64)
        self.prev_jain = np.ones(config.num_servers, dtype=np.float64)
        self.slot_index = 0
        self.current_slot = self._sample_slot()

    def reset(self) -> None:
        self.queues.fill(0.0)
        self.prices.fill(0.5 * (self.cfg.price_min + self.cfg.price_max))
        self.prev_load.fill(0.0)
        self.prev_counts.fill(0.0)
        self.prev_jain.fill(1.0)
        self.slot_index = 0
        self.current_slot = self._sample_slot()

    def clone_with_seed(self, seed: int) -> "MECEnvironment":
        return MECEnvironment(self.cfg, np.random.default_rng(seed))

    def _sample_slot(self) -> SlotSample:
        """Sample tasks and observed effective link goodputs for one slot.

        The rate samples abstract the net radio outcome.  Bandwidth,
        interference, transmit-power allocation, and handover are not separate
        controls in this computation-side environment.
        """
        scale = self.cfg.arrival_scale
        if self.cfg.arrival_distribution == "uniform":
            arrival_multiplier = np.ones(self.cfg.num_users, dtype=np.float64)
        elif self.cfg.arrival_distribution == "bursty_lognormal":
            # Mean-one multiplicative bursts change the task-size distribution
            # without changing the actor interface.  Clipping prevents a
            # single draw from dominating a finite evaluation horizon.
            sigma = 0.55
            arrival_multiplier = self.rng.lognormal(
                mean=-0.5 * sigma**2,
                sigma=sigma,
                size=self.cfg.num_users,
            )
            arrival_multiplier = np.clip(arrival_multiplier, 0.25, 3.0)
        else:
            raise ValueError(
                f"Unknown arrival distribution '{self.cfg.arrival_distribution}'."
            )

        def sample_goodput(minimum: float, maximum: float) -> np.ndarray:
            shape = (self.cfg.num_users, self.cfg.num_servers)
            if self.cfg.channel_distribution == "uniform":
                fraction = self.rng.uniform(0.0, 1.0, size=shape)
            elif self.cfg.channel_distribution == "beta_low":
                # A left-skewed, lower-goodput distribution is deliberately
                # absent from training and is used only in the OOD protocol.
                fraction = self.rng.beta(2.0, 5.0, size=shape)
            else:
                raise ValueError(
                    f"Unknown channel distribution '{self.cfg.channel_distribution}'."
                )
            return minimum + (maximum - minimum) * fraction

        return SlotSample(
            data_mb=self.rng.uniform(
                self.cfg.data_mb_min,
                self.cfg.data_mb_max,
                self.cfg.num_users,
            )
            * scale
            * arrival_multiplier,
            cycles=self.rng.uniform(
                self.cfg.cycles_min,
                self.cfg.cycles_max,
                self.cfg.num_users,
            )
            * scale
            * arrival_multiplier,
            latency_tol_s=self.rng.uniform(self.cfg.latency_min_s, self.cfg.latency_max_s, self.cfg.num_users),
            uplink_mbps=sample_goodput(
                self.cfg.uplink_min_mbps,
                self.cfg.uplink_max_mbps,
            ),
            downlink_mbps=sample_goodput(
                self.cfg.downlink_min_mbps,
                self.cfg.downlink_max_mbps,
            ),
        )

    def _queue_ratio(self, visible: bool) -> np.ndarray:
        """Return the observed queue state; no learned queue estimator is used."""
        ratio = np.clip(self.queues / self.cfg.queue_capacity_cycles, 0.0, 1.0)
        return ratio if visible else np.zeros_like(ratio)

    def _apply_price_projection(self, prices: np.ndarray) -> np.ndarray:
        """Project finite price proposals onto the configured price interval."""
        proposed = np.asarray(prices, dtype=np.float64)
        if proposed.shape != (self.cfg.num_servers,):
            raise ValueError(
                f"Expected {self.cfg.num_servers} server prices, got shape {proposed.shape}."
            )
        if not np.all(np.isfinite(proposed)):
            raise ValueError("Server-price proposals must be finite.")
        return np.clip(proposed, self.cfg.price_min, self.cfg.price_max)

    def _apply_allocation_control_projection(self, allocation_controls: np.ndarray) -> np.ndarray:
        """Project per-server utilization/shape controls onto their intervals."""
        proposed = np.asarray(allocation_controls, dtype=np.float64)
        expected_shape = (self.cfg.num_servers, 2)
        if proposed.shape != expected_shape:
            raise ValueError(
                f"Expected allocation controls with shape {expected_shape}, got {proposed.shape}."
            )
        if not np.all(np.isfinite(proposed)):
            raise ValueError("Allocation-control proposals must be finite.")
        projected = proposed.copy()
        projected[:, 0] = np.clip(
            projected[:, 0],
            self.cfg.cpu_utilization_min,
            1.0,
        )
        projected[:, 1] = np.clip(projected[:, 1], 0.0, self.cfg.alloc_temp_max)
        return projected

    def _apply_admission_projection(
        self,
        tentative_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project tentative offloading requests onto the finite-buffer set.

        Requests are considered in a deterministic rotating round-robin order.
        A rejected request is redirected to local execution before it enters a
        server queue, so rejection never represents an unreported packet drop.
        The returned per-user workload-ahead vector records earlier admitted
        work in the same slot.  It is used together with the carried backlog to
        make the arrival-before-service timing convention explicit in delay
        accounting.
        """
        executed_actions = tentative_actions.copy()
        admission = np.zeros(self.cfg.num_users, dtype=np.int8)
        redirected_cycles = np.zeros(self.cfg.num_servers, dtype=np.float64)
        same_slot_workload_ahead = np.zeros(self.cfg.num_users, dtype=np.float64)
        # Under the arrival-before-service convention, admitted work must fit
        # in the physical buffer before any current-slot service is credited.
        # This is stricter than reserving capacity expected to be released
        # later in the slot and prevents transient pre-service overflow.
        admission_budget = np.maximum(
            self.cfg.queue_capacity_cycles - self.queues,
            0.0,
        )

        for user in range(self.cfg.num_users):
            action = int(executed_actions[user])
            if action == 0:
                continue
            if action < 0:
                executed_actions[user] = 0
                continue
            server = action - 1
            if server >= self.cfg.num_servers or self.candidate_mask[user, server] < 0.5:
                executed_actions[user] = 0

        for server in range(self.cfg.num_servers):
            requested = np.where(executed_actions == (server + 1))[0]
            if requested.size == 0:
                continue
            start = (self.slot_index + server) % self.cfg.num_users
            ordered = sorted(
                requested.tolist(),
                key=lambda user: (user - start) % self.cfg.num_users,
            )
            remaining = float(admission_budget[server])
            admitted_ahead = 0.0
            for user in ordered:
                workload = float(self.current_slot.cycles[user])
                if workload <= remaining + 1.0e-9:
                    admission[user] = 1
                    same_slot_workload_ahead[user] = admitted_ahead
                    admitted_ahead += workload
                    remaining -= workload
                else:
                    executed_actions[user] = 0
                    redirected_cycles[server] += workload

        return (
            executed_actions,
            admission,
            admission_budget,
            redirected_cycles,
            same_slot_workload_ahead,
        )

    def _predicted_load(self) -> np.ndarray:
        """Return full-state workload proxy used by strong online baselines.

        This helper is intentionally not part of the decentralized server
        pricing observation. DPP-JPO and SG-JPO are full-current-state online
        benchmarks, as documented in the experiment protocol.
        """
        weighted = (
            self.current_slot.cycles[:, None] * self.candidate_mask
        ) / np.maximum(self.candidate_mask.sum(axis=1, keepdims=True), 1.0)
        return weighted.sum(axis=0)

    def get_price_observations(self, spec: MethodSpec) -> np.ndarray:
        """Return information locally available before the server broadcasts price.

        Current user task attributes are deliberately excluded because users
        have not issued their tentative requests at this point in the slot.
        """
        queue_ratio = self._queue_ratio(spec.queue_visible)
        price_ratio = (self.prices - self.cfg.price_min) / (self.cfg.price_max - self.cfg.price_min)
        count_ratio = self.prev_counts / max(self.cfg.num_users, 1)
        prev_load_ratio = np.clip(self.prev_load / self.cfg.server_cpu_hz, 0.0, 3.0)
        return np.stack(
            [
                queue_ratio,
                prev_load_ratio,
                price_ratio,
                count_ratio,
                self.prev_jain,
            ],
            axis=1,
        ).astype(np.float32)

    def get_user_observations(self, prices: np.ndarray, spec: MethodSpec) -> np.ndarray:
        slot = self.current_slot
        q_ratio = self._queue_ratio(spec.queue_visible)
        price_ratio = (prices - self.cfg.price_min) / (self.cfg.price_max - self.cfg.price_min)
        obs = []
        for user in range(self.cfg.num_users):
            candidates = self.candidate_mask[user]
            item = [
                slot.data_mb[user] / self.cfg.data_mb_max,
                slot.cycles[user] / self.cfg.cycles_max,
                slot.latency_tol_s[user] / self.cfg.latency_max_s,
                self.local_cpu_hz[user] / self.cfg.local_cpu_max_hz,
            ]
            # Fixed-size blocks are retained for parameter sharing, but a user
            # receives link and advertisement features only from candidate
            # servers.  Non-candidate entries are explicitly zeroed.
            item.extend((slot.uplink_mbps[user] / self.cfg.uplink_max_mbps) * candidates)
            item.extend((slot.downlink_mbps[user] / self.cfg.downlink_max_mbps) * candidates)
            item.extend(price_ratio * candidates)
            item.extend(q_ratio * candidates)
            item.extend(candidates)
            obs.append(item)
        return np.asarray(obs, dtype=np.float32)

    def get_allocation_observations(
        self,
        user_actions: np.ndarray,
        prices: np.ndarray,
        spec: MethodSpec,
    ) -> np.ndarray:
        slot = self.current_slot
        queue_ratio = self._queue_ratio(spec.queue_visible)
        load = np.zeros(self.cfg.num_servers, dtype=np.float64)
        count = np.zeros(self.cfg.num_servers, dtype=np.float64)
        mean_urgency = np.zeros(self.cfg.num_servers, dtype=np.float64)
        max_urgency = np.zeros(self.cfg.num_servers, dtype=np.float64)
        for server in range(self.cfg.num_servers):
            assigned = np.where(user_actions == (server + 1))[0]
            if assigned.size == 0:
                continue
            cycles = slot.cycles[assigned]
            load[server] = cycles.sum()
            count[server] = assigned.size
            uplink = slot.data_mb[assigned] * 8.0 / slot.uplink_mbps[assigned, server]
            downlink = self.cfg.output_ratio * slot.data_mb[assigned] * 8.0 / slot.downlink_mbps[assigned, server]
            urgency = (cycles / 1.0e9) / np.maximum(slot.latency_tol_s[assigned] - uplink - downlink, 0.05)
            mean_urgency[server] = float(np.mean(urgency))
            max_urgency[server] = float(np.max(urgency))
        return np.stack(
            [
                queue_ratio,
                np.clip(load / self.cfg.server_cpu_hz, 0.0, 3.0),
                (prices - self.cfg.price_min) / (self.cfg.price_max - self.cfg.price_min),
                count / max(self.cfg.num_users, 1),
                self.prev_jain,
                np.clip(mean_urgency / 8.0, 0.0, 3.0),
                np.clip(max_urgency / 8.0, 0.0, 3.0),
            ],
            axis=1,
        ).astype(np.float32)

    def get_global_state(self, prices: np.ndarray, spec: MethodSpec) -> np.ndarray:
        slot = self.current_slot
        q_ratio = self._queue_ratio(spec.queue_visible)
        price_ratio = (prices - self.cfg.price_min) / (self.cfg.price_max - self.cfg.price_min)
        server_part = np.concatenate(
            [
                q_ratio,
                price_ratio,
                np.clip(self.prev_load / self.cfg.server_cpu_hz, 0.0, 3.0),
                self.prev_jain,
            ]
        )
        user_part = np.concatenate(
            [
                slot.data_mb / self.cfg.data_mb_max,
                slot.cycles / self.cfg.cycles_max,
                slot.latency_tol_s / self.cfg.latency_max_s,
                self.local_cpu_hz / self.cfg.local_cpu_max_hz,
                slot.uplink_mbps.reshape(-1) / self.cfg.uplink_max_mbps,
                slot.downlink_mbps.reshape(-1) / self.cfg.downlink_max_mbps,
                self.candidate_mask.reshape(-1),
            ]
        )
        return np.concatenate([user_part, server_part]).astype(np.float32)

    def _local_metrics(self) -> tuple[np.ndarray, np.ndarray]:
        delay = self.current_slot.cycles / self.local_cpu_hz
        energy = self.cfg.energy_coeff * (self.local_cpu_hz ** 2) * self.current_slot.cycles
        return delay, energy

    def _offload_energy(self, users: np.ndarray, server: int) -> np.ndarray:
        data = self.current_slot.data_mb[users]
        uplink = data * 8.0 / self.current_slot.uplink_mbps[users, server]
        downlink = self.cfg.output_ratio * data * 8.0 / self.current_slot.downlink_mbps[users, server]
        return self.cfg.tx_power_w * (uplink + downlink)

    def _allocation_from_controls(
        self,
        users: np.ndarray,
        server: int,
        utilization: float,
        temperature: float,
    ) -> np.ndarray:
        """Map fixed-dimensional controls to feasible per-user CPU shares.

        ``utilization`` selects the aggregate service rate and ``temperature``
        selects how that rate is distributed across the active users.  Keeping
        these controls separate prevents the queue departure from collapsing
        to an allocation-independent full-capacity constant.
        """
        if users.size == 0:
            return np.zeros(0, dtype=np.float64)
        utilization = float(np.clip(utilization, self.cfg.cpu_utilization_min, 1.0))
        temperature = float(np.clip(temperature, 0.0, self.cfg.alloc_temp_max))
        cycles = self.current_slot.cycles[users] / self.cfg.cycles_max
        uplink = self.current_slot.data_mb[users] * 8.0 / self.current_slot.uplink_mbps[users, server]
        downlink = self.cfg.output_ratio * self.current_slot.data_mb[users] * 8.0 / self.current_slot.downlink_mbps[users, server]
        slack = np.maximum(self.current_slot.latency_tol_s[users] - uplink - downlink, 0.05)
        urgency = (self.current_slot.cycles[users] / 1.0e9) / slack
        urgency = (urgency - urgency.min()) / (urgency.max() - urgency.min() + 1.0e-6)
        score = np.exp(temperature * (0.82 * urgency + 0.18 * cycles))
        # Exact normalization preserves the selected aggregate rate.  The
        # utilization action changes queue service, whereas the temperature
        # action changes per-user execution times and Jain fairness.
        aggregate_rate = utilization * self.cfg.server_cpu_hz
        return aggregate_rate * score / score.sum()

    def _queue_transition(
        self,
        backlog: np.ndarray,
        arrivals: np.ndarray,
        allocated_cpu_rates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the allocation-coupled workload-queue recurrence.

        ``allocated_cpu_rates[s]`` is the sum of the projected CPU shares of
        the workloads served by server ``s``.  Keeping it as an explicit input
        prevents the queue update from silently bypassing the allocation
        decision: the learned utilization control changes this sum within the
        physical server-capacity bound.
        """
        backlog = np.asarray(backlog, dtype=np.float64)
        arrivals = np.asarray(arrivals, dtype=np.float64)
        allocated_cpu_rates = np.asarray(allocated_cpu_rates, dtype=np.float64)
        expected_shape = (self.cfg.num_servers,)
        for name, value in (
            ("backlog", backlog),
            ("arrivals", arrivals),
            ("allocated_cpu_rates", allocated_cpu_rates),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"Expected {name} shape {expected_shape}, got {value.shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values.")
        if np.any(backlog < 0.0) or np.any(arrivals < 0.0):
            raise ValueError("Backlog and admitted arrivals must be nonnegative.")
        if np.any(allocated_cpu_rates < 0.0):
            raise ValueError("Allocated CPU rates must be nonnegative.")
        if np.any(allocated_cpu_rates > self.cfg.server_cpu_hz + 1.0e-3):
            raise ValueError("Allocated CPU rate exceeds the physical server capacity.")

        pre_service_workload = backlog + arrivals
        serviceable = allocated_cpu_rates * self.cfg.slot_duration_s
        served_workload = np.minimum(pre_service_workload, serviceable)
        next_queues = np.maximum(pre_service_workload - served_workload, 0.0)
        return served_workload, next_queues

    def step(
        self,
        user_actions: np.ndarray,
        prices: np.ndarray,
        allocation_controls: np.ndarray,
        spec: MethodSpec,
    ) -> dict[str, Any]:
        proposed_prices = np.asarray(prices, dtype=np.float64).copy()
        proposed_allocation_controls = np.asarray(
            allocation_controls,
            dtype=np.float64,
        ).copy()
        prices = self._apply_price_projection(proposed_prices)
        allocation_controls = self._apply_allocation_control_projection(
            proposed_allocation_controls
        )
        cpu_utilizations = allocation_controls[:, 0]
        allocation_temperatures = allocation_controls[:, 1]
        tentative_actions = user_actions.copy()
        (
            effective_actions,
            admission,
            admission_budget,
            redirected_cycles,
            same_slot_workload_ahead,
        ) = self._apply_admission_projection(tentative_actions)

        slot = self.current_slot
        local_delay, local_energy = self._local_metrics()
        actual_delay = local_delay.copy()
        perceived_delay = local_delay.copy()
        energy = local_energy.copy()
        payment = np.zeros(self.cfg.num_users, dtype=np.float64)
        uplink_by_user = np.zeros(self.cfg.num_users, dtype=np.float64)
        execution_by_user = local_delay.copy()
        downlink_by_user = np.zeros(self.cfg.num_users, dtype=np.float64)
        backlog_before = self.queues.copy()
        arrivals = np.zeros(self.cfg.num_servers, dtype=np.float64)
        server_jain = np.ones(self.cfg.num_servers, dtype=np.float64)
        waiting_by_user = np.zeros(self.cfg.num_users, dtype=np.float64)
        revenue = np.zeros(self.cfg.num_servers, dtype=np.float64)
        user_cpu_allocations = np.zeros(self.cfg.num_users, dtype=np.float64)
        allocated_cpu_rates = np.zeros(self.cfg.num_servers, dtype=np.float64)

        for server in range(self.cfg.num_servers):
            assigned = np.where(effective_actions == (server + 1))[0]
            if assigned.size == 0:
                continue
            arrivals[server] = slot.cycles[assigned].sum()
            freqs = self._allocation_from_controls(
                assigned,
                server,
                float(cpu_utilizations[server]),
                float(allocation_temperatures[server]),
            )
            user_cpu_allocations[assigned] = freqs
            allocated_cpu_rates[server] = float(freqs.sum())
            jain = (freqs.sum() ** 2) / (assigned.size * np.square(freqs).sum() + 1.0e-9)
            server_jain[server] = float(jain)

            uplink = slot.data_mb[assigned] * 8.0 / slot.uplink_mbps[assigned, server]
            downlink = self.cfg.output_ratio * slot.data_mb[assigned] * 8.0 / slot.downlink_mbps[assigned, server]
            execution = slot.cycles[assigned] / freqs
            # The slot uses an arrival-before-service convention.  Besides the
            # carried backlog, a newly admitted task waits behind the work
            # admitted earlier in the same rotating order.  Its own execution
            # time still depends on its individual projected CPU share.
            aggregate_rate = max(allocated_cpu_rates[server], 1.0e-12)
            workload_ahead = backlog_before[server] + same_slot_workload_ahead[assigned]
            wait = workload_ahead / aggregate_rate
            waiting_by_user[assigned] = wait

            actual_delay[assigned] = uplink + wait + execution + downlink
            perceived_delay[assigned] = actual_delay[assigned] if spec.queue_in_delay else uplink + execution + downlink
            uplink_by_user[assigned] = uplink
            execution_by_user[assigned] = execution
            downlink_by_user[assigned] = downlink
            energy[assigned] = self._offload_energy(assigned, server)
            payment[assigned] = prices[server] * (slot.cycles[assigned] / 1.0e9)
            revenue[server] = payment[assigned].sum()

        delay_ratio = perceived_delay / np.maximum(slot.latency_tol_s, 1.0e-6)
        tail_risk = np.square(np.clip(delay_ratio - self.cfg.risk_trigger_ratio, a_min=0.0, a_max=None))
        base_user_cost = (
            self.cfg.alpha_delay * perceived_delay
            + self.cfg.beta_energy * energy
            + self.cfg.eta_payment * payment
        )
        base_user_reward = -base_user_cost
        user_reward = base_user_reward - self.cfg.violation_penalty * tail_risk

        # If a server has only carried-over backlog in the current slot, that
        # backlog remains active and receives the aggregate rate selected by
        # the current utilization action even though no new user is assigned.
        backlog_only = (backlog_before > 0.0) & (arrivals <= 0.0)
        allocated_cpu_rates[backlog_only] = (
            cpu_utilizations[backlog_only] * self.cfg.server_cpu_hz
        )

        # The known queue recurrence is evaluated from the sum of the actual
        # projected CPU shares. Neural actors do not estimate this transition.
        served_workload, next_queues = self._queue_transition(
            backlog_before,
            arrivals,
            allocated_cpu_rates,
        )
        overflow_count = int(np.count_nonzero(next_queues > self.cfg.queue_capacity_cycles + 1.0e-6))
        if overflow_count:
            raise RuntimeError("Admission projection failed to enforce the finite-buffer constraint.")
        self.queues = np.minimum(next_queues, self.cfg.queue_capacity_cycles)
        server_utility = (
            revenue
            - spec.fairness_penalty * (1.0 - server_jain)
            - spec.queue_penalty * self.queues / self.cfg.queue_capacity_cycles
        )
        self.prices = prices.copy()
        self.prev_load = arrivals
        self.prev_counts = np.array([(effective_actions == (server + 1)).sum() for server in range(self.cfg.num_servers)], dtype=np.float64)
        self.prev_jain = server_jain
        self.slot_index += 1
        self.current_slot = self._sample_slot()

        offloaded = effective_actions > 0
        requested_offloads = tentative_actions > 0
        fallback_count = int(np.count_nonzero(requested_offloads & ~offloaded))
        requested_count = int(np.count_nonzero(requested_offloads))
        offloaded_count = int(np.count_nonzero(offloaded))
        fallback_ratio = fallback_count / max(requested_count, 1)
        acceptance_ratio = offloaded_count / max(requested_count, 1)
        avg_wait = float(waiting_by_user[offloaded].mean()) if offloaded.any() else 0.0
        conditional_payment = float(payment[offloaded].mean()) if offloaded.any() else 0.0
        max_queue_occupancy = float(
            np.max(np.maximum(backlog_before, self.queues) / self.cfg.queue_capacity_cycles)
        )

        effective_request_violation = False
        for user, action in enumerate(effective_actions):
            if action < 0 or action > self.cfg.num_servers:
                effective_request_violation = True
                break
            if action > 0 and self.candidate_mask[user, action - 1] < 0.5:
                effective_request_violation = True
                break
        hard_constraint_violation = bool(
            overflow_count
            or np.any(prices < self.cfg.price_min - 1.0e-12)
            or np.any(prices > self.cfg.price_max + 1.0e-12)
            or np.any(allocation_controls[:, 0] < self.cfg.cpu_utilization_min - 1.0e-12)
            or np.any(allocation_controls[:, 0] > 1.0 + 1.0e-12)
            or np.any(allocation_controls[:, 1] < -1.0e-12)
            or np.any(allocation_controls[:, 1] > self.cfg.alloc_temp_max + 1.0e-12)
            or np.any(allocated_cpu_rates > self.cfg.server_cpu_hz + 1.0e-3)
            or effective_request_violation
        )
        request_interventions = int(np.count_nonzero(tentative_actions != effective_actions))
        continuous_interventions = int(
            np.count_nonzero(~np.isclose(proposed_prices, prices))
            + np.count_nonzero(
                ~np.isclose(proposed_allocation_controls, allocation_controls)
            )
        )
        intervention_denominator = self.cfg.num_users + 3 * self.cfg.num_servers
        metrics = {
            "avg_delay": float(actual_delay.mean()),
            "p95_delay": float(np.quantile(actual_delay, 0.95)),
            "avg_uplink_delay": float(uplink_by_user.mean()),
            "avg_waiting_delay_all_users": float(waiting_by_user.mean()),
            "avg_execution_delay": float(execution_by_user.mean()),
            "avg_downlink_delay": float(downlink_by_user.mean()),
            "avg_energy": float(energy.mean()),
            "avg_payment": float(payment.mean()),
            "avg_payment_if_offloaded": conditional_payment,
            "avg_server_price": float(prices.mean()),
            "avg_profit": float(server_utility.mean()),
            "fairness": float(server_jain.mean()),
            "avg_queue_backlog": float(backlog_before.mean()),
            "avg_waiting_delay": avg_wait,
            "violation_ratio": float((actual_delay > slot.latency_tol_s).mean()),
            "requested_offload_ratio": float(requested_count / self.cfg.num_users),
            "edge_participation_ratio": float(offloaded_count / self.cfg.num_users),
            "local_execution_ratio": float(1.0 - offloaded_count / self.cfg.num_users),
            "admission_acceptance_ratio": float(acceptance_ratio),
            "max_queue_occupancy": max_queue_occupancy,
            "fallback_ratio": float(fallback_ratio),
            "redirected_workload": float(redirected_cycles.sum()),
            "overflow_count": float(overflow_count),
            "hard_constraint_violation_rate": float(hard_constraint_violation),
            "projection_intervention_ratio": float(
                (request_interventions + continuous_interventions)
                / max(intervention_denominator, 1)
            ),
            "avg_cpu_utilization": float(
                np.mean(allocated_cpu_rates / self.cfg.server_cpu_hz)
            ),
        }
        return {
            "user_reward_mean": float(user_reward.mean()),
            "base_user_reward_mean": float(base_user_reward.mean()),
            "constraint_cost_mean": float(
                (actual_delay > slot.latency_tol_s).mean()
            ),
            "server_reward_mean": float(server_utility.mean()),
            "tentative_actions": tentative_actions,
            "executed_actions": effective_actions,
            "admission_indicators": admission,
            "admission_budget": admission_budget,
            "same_slot_workload_ahead": same_slot_workload_ahead,
            "waiting_by_user": waiting_by_user,
            "actual_delay_by_user": actual_delay,
            "uplink_delay_by_user": uplink_by_user,
            "execution_delay_by_user": execution_by_user,
            "downlink_delay_by_user": downlink_by_user,
            "energy_by_user": energy,
            "executed_prices": prices.copy(),
            "executed_allocation_controls": allocation_controls.copy(),
            "executed_cpu_utilizations": cpu_utilizations.copy(),
            "executed_allocation_temperatures": allocation_temperatures.copy(),
            "user_cpu_allocations": user_cpu_allocations,
            "allocated_cpu_rates": allocated_cpu_rates,
            "served_workload": served_workload,
            "user_payment": payment,
            "offloaded_indicators": offloaded.astype(np.float64),
            "requested_offload_indicators": requested_offloads.astype(np.float64),
            "violation_indicators": (
                actual_delay > slot.latency_tol_s
            ).astype(np.float64),
            "metrics": metrics,
        }
