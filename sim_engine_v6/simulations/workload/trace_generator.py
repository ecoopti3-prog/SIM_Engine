"""
trace_generator.py — AI Workload Dynamic Trace Generator

Real AI inference is NOT 400W constant. It's:
  IDLE → DRAM_FETCH → COMPUTE_GEMM → ACTIVATE → ATTENTION → SOFTMAX → IDLE → ...

Each phase has a distinct power envelope and di/dt profile.
The IDLE→COMPUTE transition is when PDN droop and thermal transients peak.

This module generates P(t) traces for:
  - LLM Inference (token-by-token, memory-bound)
  - Training (compute-bound, sustained GEMM)
  - Attention (hybrid: memory + compute)

Outputs: WorkloadTrace with per-phase power, di/dt, and time vectors
for injection into thermal and PDN simulators.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class WorkloadPhase:
    name: str
    power_fraction: float      # fraction of TDP (0-1.0, can exceed 1.0 for burst)
    duration_us: float         # duration in microseconds
    di_dt_scale: float         # relative di/dt at entry (1.0 = nominal)
    description: str = ""


# ── Canonical AI workload phase libraries ─────────────────────────────────────
# Based on published GPU profiling data (A100, H100, Groq LPU traces)

WORKLOAD_PROFILES: Dict[str, List[WorkloadPhase]] = {

    "llm_inference_prefill": [
        # Prefill (prompt processing) — memory-bound, then compute burst
        WorkloadPhase("dram_fetch_kv",    0.45, 8.0,  0.3, "KV cache DRAM fetch"),
        WorkloadPhase("compute_qk",       0.95, 2.0,  1.8, "Q×K matrix multiply, full ALU burst"),
        WorkloadPhase("softmax",          0.50, 1.0,  0.7, "Attention softmax normalization"),
        WorkloadPhase("compute_av",       0.90, 2.0,  1.5, "A×V projection, high compute"),
        WorkloadPhase("linear_proj",      0.85, 1.5,  1.3, "FFN linear projection"),
        WorkloadPhase("gelu_act",         0.40, 0.5,  0.6, "GELU activation, low power"),
        WorkloadPhase("linear_proj2",     0.85, 1.5,  1.3, "FFN second projection"),
        WorkloadPhase("layernorm",        0.30, 0.5,  0.4, "LayerNorm, memory bandwidth"),
        WorkloadPhase("idle_inter_layer", 0.05, 1.0,  0.0, "Pipeline stall between layers"),
    ],

    "llm_inference_decode": [
        # Decode (autoregressive) — extremely memory-bound, lots of idle
        WorkloadPhase("dram_fetch_kv",    0.60, 20.0, 0.2, "KV cache fetch — dominant cost"),
        WorkloadPhase("compute_qk",       0.80, 0.5,  2.0, "Single-token Q×K, short burst"),
        WorkloadPhase("softmax",          0.35, 0.2,  0.5, "Softmax over full context"),
        WorkloadPhase("compute_av",       0.75, 0.5,  1.4, "Single-token A×V"),
        WorkloadPhase("linear_proj",      0.70, 0.3,  1.2, "FFN with tiny batch"),
        WorkloadPhase("idle_wait",        0.04, 5.0,  0.0, "Waiting for next token request"),
    ],

    "training_fwd_bwd": [
        # Training: sustained heavy compute, high power, high di/dt on grad sync
        WorkloadPhase("data_load",        0.35, 5.0,  0.2, "DataLoader prefetch"),
        WorkloadPhase("fwd_compute",      0.98, 15.0, 1.6, "Forward pass — full GEMM utilization"),
        WorkloadPhase("loss_compute",     0.55, 1.0,  0.8, "Loss + reduction ops"),
        WorkloadPhase("bwd_compute",      1.05, 18.0, 1.8, "Backward pass — peak power (>TDP)"),
        WorkloadPhase("grad_allreduce",   0.70, 8.0,  1.1, "All-reduce gradient sync"),
        WorkloadPhase("optimizer_step",   0.60, 3.0,  0.9, "AdamW update step"),
        WorkloadPhase("idle_sync",        0.08, 2.0,  0.0, "Barrier sync wait"),
    ],

    "custom_sustained": [
        # Simple high-utilization sustained workload
        WorkloadPhase("warmup",     0.30, 2.0,  0.5, "Clock warmup"),
        WorkloadPhase("ramp_up",    0.80, 1.0,  2.5, "Fast ramp to full load"),
        WorkloadPhase("sustained",  0.95, 50.0, 0.1, "Sustained compute"),
        WorkloadPhase("cooldown",   0.20, 2.0,  0.3, "Ramp down"),
    ],
}


@dataclass
class WorkloadTrace:
    """
    Time-domain power trace for one workload cycle.
    All arrays are synchronized — same length, same time axis.
    """
    profile_name: str
    tdp_w: float

    # Time axis [µs]
    time_us: np.ndarray = field(default_factory=lambda: np.array([]))

    # Power envelope [W] — actual instantaneous power
    power_w: np.ndarray = field(default_factory=lambda: np.array([]))

    # di/dt [A/ns] — rate of current change at each timestep
    di_dt: np.ndarray = field(default_factory=lambda: np.array([]))

    # Phase labels per timestep (for visualization)
    phase_labels: List[str] = field(default_factory=list)

    # Derived statistics
    p_avg_w: float = 0.0
    p_peak_w: float = 0.0
    p_idle_w: float = 0.0
    peak_di_dt: float = 0.0          # peak di/dt [A/ns]
    burst_count: int = 0             # number of idle→compute transitions
    duty_cycle_pct: float = 0.0      # fraction of time above 50% TDP
    total_duration_us: float = 0.0

    def to_thermal_input(self, sample_rate_hz: float = 1e6) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample trace to thermal timescale (microseconds → milliseconds)."""
        # Thermal time constant is ~ms; resample to 1kHz
        t_ms = self.time_us * 1e-3
        factor = max(1, int(len(self.time_us) / 1000))
        return t_ms[::factor], self.power_w[::factor]

    def worst_case_di_dt(self) -> float:
        """Return peak di/dt for PDN simulation."""
        return float(self.peak_di_dt)

    def burst_power_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (time_ns, power_w) focused on first burst event for PDN droop."""
        # Find first idle→compute transition
        t_ns = self.time_us * 1e3  # µs → ns
        for i in range(1, len(self.power_w)):
            if self.power_w[i-1] < self.tdp_w * 0.1 and self.power_w[i] > self.tdp_w * 0.5:
                # Found transition: return ±50ns window
                start = max(0, i - 5)
                end   = min(len(self.power_w), i + 100)
                return t_ns[start:end] - t_ns[start], self.power_w[start:end]
        return t_ns[:100], self.power_w[:100]


def generate_trace(
    profile_name: str = "llm_inference_prefill",
    tdp_w: float = 400.0,
    vdd_v: float = 0.85,
    n_cycles: int = 2,
    dt_ns: float = 1.0,        # simulation timestep [ns]
    rng_seed: Optional[int] = 42,
) -> WorkloadTrace:
    """
    Generate a realistic P(t) workload trace.

    Args:
        profile_name: one of WORKLOAD_PROFILES keys
        tdp_w: thermal design power [W]
        vdd_v: supply voltage for I = P/V calculation
        n_cycles: how many full workload cycles to simulate
        dt_ns: timestep in nanoseconds
        rng_seed: for reproducibility (None = random)
    """
    rng = np.random.default_rng(rng_seed)
    profile = WORKLOAD_PROFILES.get(profile_name, WORKLOAD_PROFILES["llm_inference_prefill"])

    # Build time-domain trace
    time_pts     = []
    power_pts    = []
    didt_pts     = []
    phase_labels = []

    t_current_ns  = 0.0
    prev_power    = tdp_w * profile[0].power_fraction

    for _cycle in range(n_cycles):
        for phase in profile:
            duration_ns  = phase.duration_us * 1000.0  # µs → ns
            n_steps      = max(2, int(duration_ns / dt_ns))
            target_power = tdp_w * phase.power_fraction

            # Ramp shape: fast rise (5% of duration) then flat with noise
            ramp_steps = max(1, int(n_steps * 0.05))

            for i in range(n_steps):
                t = t_current_ns + i * dt_ns

                # Ramp from prev to target
                if i < ramp_steps:
                    frac = i / ramp_steps
                    p_base = prev_power + frac * (target_power - prev_power)
                else:
                    p_base = target_power

                # Add realistic microarchitectural noise (cache misses, pipeline stalls)
                noise_sigma = tdp_w * 0.04  # 4% TDP variation
                p_actual = p_base + rng.normal(0, noise_sigma)
                p_actual = max(0.0, p_actual)

                # di/dt: highest at transitions, noise during steady-state
                if i < ramp_steps:
                    dp = target_power - prev_power
                    di_dt_val = abs(dp / vdd_v) / (ramp_steps * dt_ns)  # A/ns
                    di_dt_val *= phase.di_dt_scale
                else:
                    # Stochastic SSN: cores switch randomly → some di/dt always present
                    di_dt_val = abs(rng.normal(0, tdp_w * 0.01 / vdd_v / dt_ns))

                time_pts.append(t)
                power_pts.append(p_actual)
                didt_pts.append(di_dt_val)
                phase_labels.append(phase.name)

            prev_power   = target_power
            t_current_ns += duration_ns

    time_arr  = np.array(time_pts)
    power_arr = np.array(power_pts)
    didt_arr  = np.array(didt_pts)

    # Compute statistics
    p_avg     = float(np.mean(power_arr))
    p_peak    = float(np.max(power_arr))
    p_idle    = float(tdp_w * 0.04)
    peak_didt = float(np.max(didt_arr))
    duty      = float(np.mean(power_arr > tdp_w * 0.5) * 100)

    # Count idle→compute bursts
    above_half = power_arr > tdp_w * 0.5
    bursts = int(np.sum(np.diff(above_half.astype(int)) == 1))

    trace = WorkloadTrace(
        profile_name=profile_name,
        tdp_w=tdp_w,
        time_us=time_arr * 1e-3,   # ns → µs
        power_w=power_arr,
        di_dt=didt_arr,
        phase_labels=phase_labels,
        p_avg_w=round(p_avg, 2),
        p_peak_w=round(p_peak, 2),
        p_idle_w=round(p_idle, 2),
        peak_di_dt=round(peak_didt, 4),
        burst_count=bursts,
        duty_cycle_pct=round(duty, 1),
        total_duration_us=round(float(time_arr[-1]) * 1e-3, 2),
    )
    return trace
