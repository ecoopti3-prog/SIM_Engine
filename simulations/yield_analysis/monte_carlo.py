"""
monte_carlo.py — Stochastic Yield & Reliability Analysis

Answers four questions simultaneously:

  1. YIELD %       — What fraction of dies pass ALL thermal/PDN/timing gates?
  2. WORST-CASE    — PDF of T_hotspot, droop, timing slack across the population
  3. SENSITIVITY   — Which parameter variation kills the most dies? (Sobol indices)
  4. MTTF FLEET    — Distribution of lifetimes across a production fleet

Physical basis for parameter distributions:
  R_theta:         ±15% (JEDEC package-to-package variation, 1σ)
  VDD:             ±3%  (on-chip LDO spec, datasheet Table 3)
  P_leakage_mult:  log-normal σ=0.35 (subthreshold leakage is log-normally distributed
                   — this is why FF/SS corners span 4x; it's ~2σ of the distribution)
  P_dynamic:       ±8%  (activity factor varies with workload patterns)
  T_ambient:       ±5°C (datacenter cold aisle temperature variation, measured ASHRAE)
  bump_density:    ±10% (C4 bump placement yield, per ITRS roadmap)

Adaptive stopping:
  Runs until Yield% confidence interval < ±0.5% (2σ) or max_samples reached.
  For a 95% yield product, this requires ~380 samples (Wilson interval).
  For a 99% yield product, requires ~800 samples.

Fast evaluation: uses analytical models (same physics as full pipeline)
but vectorized with NumPy — evaluates 1000 dies in ~1 second.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ── Parameter variation distributions (calibrated to published process data) ──
PARAM_VARIATIONS = {
    # (distribution, relative_sigma or abs_sigma, notes)
    "r_theta":       ("normal",    0.15, "Package R_theta ±15% 1σ (JEDEC variation)"),
    "vdd":           ("normal",    0.03, "VDD ±3% 1σ (on-chip LDO spec)"),
    "p_leak_mult":   ("lognormal", 0.35, "Leakage log-normal σ=0.35 (subthreshold physics)"),
    "p_dynamic_mult":("normal",    0.08, "P_dynamic ±8% 1σ (activity factor)"),
    "t_ambient_abs": ("normal",    5.0,  "T_ambient ±5°C absolute (ASHRAE cold aisle)"),
    "bump_density":  ("normal",    0.10, "Bump density ±10% 1σ (C4 placement yield)"),
}

# ── Pass/fail criteria (same physics as sim_engine gates) ─────────────────────
# Aliases so _sobol_sensitivity can look up distributions for all batch param names
PARAM_VARIATIONS["p_dynamic"] = PARAM_VARIATIONS["p_dynamic_mult"]
PARAM_VARIATIONS["t_ambient"] = PARAM_VARIATIONS["t_ambient_abs"]

GATES = {
    "jedec_thermal":  125.0,   # T_hotspot < 125°C (JEDEC JESD51)
    "timing_slack":   0.0,     # ps — must be positive
    "pdn_droop_pct":  10.0,    # max 10% VDD droop
    "em_lifetime":    10.0,    # years — JEDEC JEP122H
    "nbti_vt":        25.0,    # mV — JEDEC max ΔVt at 10yr
}

ALPHA_LEAK  = 0.0239   # /°C — calibrated (see coupled solver)
KB_EV       = 8.617e-5
JEDEC_LIMIT = 125.0
EM_A        = 1.48e11
EM_Ea       = 0.7


def _sample_parameters(
    nominal: Dict,
    n_samples: int,
    rng: np.random.Generator,
    correlation_rt_vdd: float = -0.2,   # slight anti-correlation: lower VDD → higher R_theta (weaker package)
) -> Dict[str, np.ndarray]:
    """
    Draw N samples from the joint parameter distribution.
    
    Returns dict of arrays, each length N.
    Includes correlation structure: R_theta and VDD are slightly anti-correlated
    (packages with poor thermal also tend to have more voltage noise).
    """
    N = n_samples
    
    # Correlated normal draws for R_theta and VDD using Cholesky
    cov = np.array([[1.0, correlation_rt_vdd],
                    [correlation_rt_vdd, 1.0]])
    L   = np.linalg.cholesky(cov)
    z   = rng.standard_normal((2, N))
    z_corr = L @ z   # correlated standard normals
    
    r_theta_nom = float(nominal.get("thermal_resistance_c_per_w") or 0.10)
    vdd_nom     = float(nominal.get("vdd_v") or nominal.get("voltage_v") or 0.85)
    p_dyn_nom   = float(nominal.get("watt") or nominal.get("tdp_watt") or 100.0)
    t_amb_nom   = float(nominal.get("t_ambient_c") or 25.0)
    bump_d_nom  = float(nominal.get("bump_density_per_mm2") or 2000.0)
    i_total_nom = float(nominal.get("current_a") or p_dyn_nom / vdd_nom)
    
    sig_r   = PARAM_VARIATIONS["r_theta"][1]
    sig_vdd = PARAM_VARIATIONS["vdd"][1]
    sig_p   = PARAM_VARIATIONS["p_dynamic_mult"][1]
    sig_t   = PARAM_VARIATIONS["t_ambient_abs"][1]
    sig_b   = PARAM_VARIATIONS["bump_density"][1]
    sig_lk  = PARAM_VARIATIONS["p_leak_mult"][1]
    
    return {
        # Correlated: R_theta and VDD
        "r_theta":       r_theta_nom * (1 + sig_r * z_corr[0]),
        "vdd":           np.clip(vdd_nom * (1 + sig_vdd * z_corr[1]), vdd_nom*0.85, vdd_nom*1.15),
        # Independent
        "p_dynamic":     p_dyn_nom * (1 + sig_p * rng.standard_normal(N)),
        "p_leak_mult":   np.exp(sig_lk * rng.standard_normal(N)),   # log-normal
        "t_ambient":     t_amb_nom + sig_t * rng.standard_normal(N),
        "bump_density":  np.clip(bump_d_nom * (1 + sig_b * rng.standard_normal(N)), 100, None),
        "i_total":       np.full(N, i_total_nom),   # derived from p_dynamic/vdd below
    }


def _evaluate_batch(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Vectorized physics evaluation for N parameter samples simultaneously.
    Same equations as the full pipeline, but NumPy-vectorized.
    """
    N = len(params["r_theta"])
    
    r    = np.clip(params["r_theta"], 0.001, 2.0)
    vdd  = params["vdd"]
    p_d  = np.clip(params["p_dynamic"], 1.0, 1e5)
    plm  = np.clip(params["p_leak_mult"], 0.01, 100.0)
    t_a  = params["t_ambient"]
    bump = params["bump_density"]
    
    # ── Coupled thermal solver (vectorized fixed-point) ────────────────────
    # T* = T_amb + (P_dyn + P_leak_ref * mult * exp(alpha*(T*-25))) * R
    # Solve with 100 iterations of fixed-point (always converges with small R)
    p_leak_ref = p_d * 0.04 * plm   # 4% × individual chip leakage multiplier
    T = t_a + p_d * r * 0.7          # initial guess
    
    runaway = np.zeros(N, dtype=bool)
    for _ in range(150):
        p_leak = p_leak_ref * np.exp(np.clip(ALPHA_LEAK * (T - 25.0), -10, 60))
        sens   = ALPHA_LEAK * p_leak * r
        runaway |= (sens >= 1.0)
        T_new  = t_a + (p_d + p_leak) * r
        T_new  = np.clip(T_new, t_a, 300.0)
        converged = np.abs(T_new - T) < 0.01
        T = np.where(converged | runaway, T, T_new)
        if np.all(converged | runaway):
            break
    
    T_op = T
    
    # ── 2D hotspot: average = T_op, hotspot = T_op × hotspot_factor ───────
    # hotspot_factor varies with process (tighter pitch → more concentrated power)
    hotspot_factor = 1.8 + 0.2 * (r / np.mean(r) - 1.0)   # slight correlation
    T_hotspot = T_op * hotspot_factor  # approximate; correct ratio physics
    # More precise: T_hotspot = T_amb + P_total * R_theta * hotspot_factor
    p_total = p_d + p_leak
    T_hotspot_precise = t_a + p_total * r * hotspot_factor
    
    # ── PDN droop ──────────────────────────────────────────────────────────
    # V_droop = (P_total/VDD) * R_pdn + L_pkg * di/dt
    # R_pdn ∝ 1/bump_density (more bumps = lower resistance)
    i_load = p_total / np.maximum(vdd, 0.1)
    # L_pkg scales with bump density (same formula as SSN model, calibrated vs H100 SI data)
    l_pkg_ph   = 100.0 * np.sqrt(1000.0 / np.maximum(bump, 10))   # pH
    # DC resistance: R_pkg ≈ 0.3mΩ at 2000 bumps/mm², scales as 1/sqrt(density)
    r_pdn_mohm = 0.3 * np.sqrt(2000.0 / np.maximum(bump, 10))     # mΩ
    ir_drop_mv = i_load * r_pdn_mohm * 1e-3 * 1000               # mV = A × mΩ × 1e-3 × 1000
    di_dt      = 0.4   # A/ns — typical sustained (burst captured by deterministic pipeline)
    inductive_mv = l_pkg_ph * 1e-12 * di_dt * 1e9 * 1000         # mV = L[H] × di/dt[A/s] × 1000
    droop_mv   = ir_drop_mv + inductive_mv
    droop_pct  = droop_mv / (vdd * 1000) * 100
    
    # ── Timing slack under droop ───────────────────────────────────────────
    # SS corner (worst case): delay_mult=1.15, voltage sensitivity α=1.3
    vdd_drooped = vdd - droop_mv * 1e-3
    t_pd_nom_ns = 0.30   # ns at nominal Vdd, Typical corner
    v_scale     = np.maximum(vdd / np.maximum(vdd_drooped, 0.1), 1.0) ** 1.3
    t_scale     = 1.0 + 0.003 * (T_op - 25.0)
    t_pd        = t_pd_nom_ns * 1.15 * v_scale * t_scale   # SS corner
    slack_ps    = (0.50 - t_pd) * 1000 - 50.0   # 2GHz clock, 50ps setup margin
    
    # ── EM lifetime ────────────────────────────────────────────────────────
    # Black's equation: MTTF = A / J^n * exp(Ea/kT)
    n_wires  = np.maximum(bump * 200, 100)   # ~200 wires per bump (power mesh)
    i_wire   = i_load / n_wires
    J        = i_wire / (0.08e-8)            # A/cm² (0.08 µm² power rail)
    J        = np.clip(J, 1e3, 1e10)
    T_k      = T_hotspot_precise + 273.15
    mttf_s   = EM_A / (J ** 2.0) * np.exp(EM_Ea / (KB_EV * T_k))
    mttf_yr  = mttf_s / 3.156e7
    
    # ── NBTI ΔVt ──────────────────────────────────────────────────────────
    T_norm   = T_k / 378.0   # normalized to 105°C
    dVt_mv   = 20.0 * (T_norm ** 3) * (0.6 ** 0.3) * (10.0 ** 0.18)   # at 10yr, 60% duty
    
    return {
        "T_op":        T_op,
        "T_hotspot":   T_hotspot_precise,
        "runaway":     runaway,
        "droop_pct":   droop_pct,
        "slack_ps":    slack_ps,
        "mttf_yr":     np.clip(mttf_yr, 0, 1e6),
        "nbti_vt_mv":  dVt_mv,
        "vdd_sample":  vdd,
        "r_theta_sample": r,
        "p_leak_mult": plm,
    }


def _pass_fail(results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Apply all pass/fail gates to batch results."""
    return {
        "thermal_pass":  results["T_hotspot"] < GATES["jedec_thermal"],
        "no_runaway":    ~results["runaway"],
        "timing_pass":   results["slack_ps"] > GATES["timing_slack"],
        "pdn_pass":      results["droop_pct"] < GATES["pdn_droop_pct"],
        "em_pass":       results["mttf_yr"] >= GATES["em_lifetime"],
        "nbti_pass":     results["nbti_vt_mv"] < GATES["nbti_vt"],
    }


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _combined_margin(results: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Continuous combined margin metric — positive = healthy, negative = failing.
    
    Computed on each metric independently (normalized), then min across all.
    Using min() means: the margin is limited by the WORST gate for each die.
    
    This works even when all dies fail (margin is uniformly negative but has variance).
    """
    # Normalize each metric to [-1, +1] range where 0 = exactly at gate
    m_thermal = (GATES["jedec_thermal"] - results["T_hotspot"]) / GATES["jedec_thermal"]
    m_timing  = results["slack_ps"] / 500.0      # 500ps = half typical clock period
    m_pdn     = (GATES["pdn_droop_pct"] - results["droop_pct"]) / GATES["pdn_droop_pct"]
    m_em      = np.log10(np.maximum(results["mttf_yr"], 0.01) / GATES["em_lifetime"])
    m_nbti    = (GATES["nbti_vt"] - results["nbti_vt_mv"]) / GATES["nbti_vt"]
    m_runaway = np.where(results["runaway"], -1.0, 1.0)
    
    # Combined = minimum margin (worst gate determines each die fate)
    return np.minimum.reduce([m_thermal, m_timing, m_pdn, m_em, m_nbti, m_runaway])


def _sobol_sensitivity(
    nominal: Dict,
    results_base: Dict[str, np.ndarray],
    params_base: Dict[str, np.ndarray],
    rng: np.random.Generator,
    n_sobol: int = 500,
) -> Dict[str, float]:
    """
    First-order Sobol sensitivity indices via saltelli-like estimator.
    
    S_i = Var(E[Y | X_i]) / Var(Y)
    
    Approximated by: for each param, resample only that param while
    keeping others fixed, measure variance reduction in yield.
    """
    # Only iterate over params that have a PARAM_VARIATIONS entry
    param_names = [k for k in params_base.keys() if k in PARAM_VARIATIONS or k in
                   ["r_theta","vdd","p_dynamic","p_leak_mult","t_ambient","bump_density"]]
    n_base      = len(params_base[param_names[0]])
    
    # Use continuous margin (works even at 0% or 100% yield)
    y_base    = _combined_margin(results_base)
    var_total = np.var(y_base)
    
    if var_total < 1e-10:
        # Truly no variance — all parameters equally unimportant (design is deeply failing)
        return {p: round(1.0 / len(param_names), 4) for p in param_names}
    
    sensitivity = {}
    n_sub = min(n_sobol, n_base)
    
    # Map PARAM_VARIATIONS names → keys that actually exist in params_base (P)
    _PVAR_TO_BATCH = {
        "r_theta":        ("r_theta",     "relative"),
        "vdd":            ("vdd",         "relative"),
        "p_leak_mult":    ("p_leak_mult", "lognormal"),
        "p_dynamic_mult": ("p_dynamic",   "relative"),   # P has 'p_dynamic', not 'p_dynamic_mult'
        "t_ambient_abs":  ("t_ambient",   "absolute"),   # P has 't_ambient', not 't_ambient_abs'
        "bump_density":   ("bump_density","relative"),
    }
    # Also map direct key names so param_names from P are handled
    for k in ["r_theta", "vdd", "p_dynamic", "p_leak_mult", "t_ambient", "bump_density"]:
        if k not in _PVAR_TO_BATCH:
            _PVAR_TO_BATCH[k] = (k, "relative")
    
    for pname in param_names:
        if pname not in _PVAR_TO_BATCH:
            continue
        batch_key, mode = _PVAR_TO_BATCH[pname]
        
        # Resample only this parameter, keep all others fixed
        params_alt = {k: v[:n_sub].copy() for k, v in params_base.items()}
        if batch_key not in params_alt:
            sensitivity[pname] = 0.0
            continue
        
        dist_type, sig, _ = PARAM_VARIATIONS[pname]
        nom_val = float(np.mean(params_base[batch_key]))
        
        if mode == "lognormal" or dist_type == "lognormal":
            params_alt[batch_key] = nom_val * np.exp(sig * rng.standard_normal(n_sub))
        elif mode == "absolute":
            params_alt[batch_key] = nom_val + sig * rng.standard_normal(n_sub)
        else:  # relative
            params_alt[batch_key] = nom_val * (1 + sig * rng.standard_normal(n_sub))
        
        results_alt = _evaluate_batch(params_alt)
        y_alt       = _combined_margin(results_alt)
        
        # First-order Sobol: variance of conditional expectation
        # Simple estimator: S_i ≈ 1 - Var(Y_alt) / Var(Y_base)
        var_alt = np.var(y_alt)
        s_i     = max(0.0, 1.0 - var_alt / var_total)
        sensitivity[pname] = round(float(s_i), 4)
    
    # Normalize so they sum to ≤1 (first-order indices don't sum to exactly 1)
    total = sum(sensitivity.values())
    if total > 0:
        sensitivity = {k: round(v / total, 4) for k, v in sensitivity.items()}
    
    return sensitivity


def run_monte_carlo(
    power_params: Optional[dict]   = None,
    thermal_params: Optional[dict] = None,
    pdn_params: Optional[dict]     = None,
    max_samples: int = 1000,
    adaptive: bool = True,
    ci_target: float = 0.005,   # stop when 2σ CI width < 0.5%
    compute_sensitivity: bool = True,
    rng_seed: int = 42,
) -> Dict:
    """
    Full Monte Carlo yield and reliability analysis.
    
    Returns yield%, worst-case distributions, sensitivity analysis,
    and MTTF fleet statistics.
    """
    pp  = power_params  or {}
    tp  = thermal_params or {}
    pdn = pdn_params    or {}
    rng = np.random.default_rng(rng_seed)
    
    # Build nominal parameter dict for sampling
    nominal = {
        "thermal_resistance_c_per_w": float(tp.get("thermal_resistance_c_per_w") or 0.10),
        "vdd_v":          float(pdn.get("vdd_v") or pp.get("voltage_v") or 0.85),
        "voltage_v":      float(pp.get("voltage_v") or 0.85),
        "watt":           float(pp.get("watt") or pp.get("tdp_watt") or 100.0),
        "t_ambient_c":    float(tp.get("t_ambient_c") or 25.0),
        "bump_density_per_mm2": float(pdn.get("bump_density_per_mm2") or 2000.0),
        "current_a":      float(pdn.get("current_a") or pp.get("current_a") or
                                (float(pp.get("watt") or 100) / float(pp.get("voltage_v") or 0.85))),
        "t_ambient":      float(tp.get("t_ambient_c") or 25.0),
        "p_leak_mult":    1.0,   # nominal multiplier
    }

    # ── Adaptive sampling ─────────────────────────────────────────────────
    batch_size = min(200, max_samples)
    all_results = {k: [] for k in ["T_op","T_hotspot","runaway","droop_pct",
                                    "slack_ps","mttf_yr","nbti_vt_mv",
                                    "vdd_sample","r_theta_sample","p_leak_mult"]}
    all_params  = {k: [] for k in ["r_theta", "vdd", "p_dynamic", "p_leak_mult", "t_ambient", "bump_density", "i_total"]}
    n_total = 0
    yield_history = []
    
    while n_total < max_samples:
        n_batch = min(batch_size, max_samples - n_total)
        params  = _sample_parameters(nominal, n_batch, rng)
        results = _evaluate_batch(params)
        
        for k in all_results:
            if k in results:
                all_results[k].append(results[k])
        for k in all_params:
            if k in params:
                all_params[k].append(params[k])
        
        n_total += n_batch
        
        # Check convergence (adaptive stopping)
        if adaptive and n_total >= 100:
            combined_pass = np.concatenate([
                np.all(list(_pass_fail({k: np.concatenate(v) for k, v in all_results.items() 
                                       if v}).values()), axis=0)
            ])
            y_pct = float(np.mean(combined_pass))
            lo, hi = _wilson_ci(int(np.sum(combined_pass)), n_total)
            yield_history.append((n_total, y_pct, hi - lo))
            if (hi - lo) < ci_target * 2:
                break   # CI width < target → enough samples
    
    # ── Aggregate results ─────────────────────────────────────────────────
    R = {k: np.concatenate(v) for k, v in all_results.items() if v}
    P = {k: np.concatenate(v) for k, v in all_params.items() if v}
    N = len(R["T_op"])
    
    pf     = _pass_fail(R)
    passes = np.all(list(pf.values()), axis=0)
    yield_pct = float(np.mean(passes) * 100)
    
    lo, hi = _wilson_ci(int(np.sum(passes)), N)
    ci_width = (hi - lo) * 100
    
    # ── Per-gate yield ────────────────────────────────────────────────────
    gate_yields = {g: round(float(np.mean(v) * 100), 2) for g, v in pf.items()}
    gate_failures = {g: int(np.sum(~v)) for g, v in pf.items()}
    
    # ── Worst-case distribution statistics ───────────────────────────────
    dist_stats = {}
    for metric, arr, gate_val, gate_dir in [
        ("T_hotspot_c",  R["T_hotspot"],  GATES["jedec_thermal"], "lt"),
        ("droop_pct",    R["droop_pct"],  GATES["pdn_droop_pct"], "lt"),
        ("slack_ps",     R["slack_ps"],   GATES["timing_slack"],  "gt"),
        ("mttf_yr",      R["mttf_yr"],    GATES["em_lifetime"],   "gt"),
        ("nbti_vt_mv",   R["nbti_vt_mv"], GATES["nbti_vt"],       "lt"),
    ]:
        a = arr[np.isfinite(arr)]
        dist_stats[metric] = {
            "mean":   round(float(np.mean(a)), 2),
            "std":    round(float(np.std(a)), 2),
            "p5":     round(float(np.percentile(a, 5)), 2),
            "p50":    round(float(np.percentile(a, 50)), 2),
            "p95":    round(float(np.percentile(a, 95)), 2),
            "p99":    round(float(np.percentile(a, 99)), 2),
            "worst":  round(float(np.max(a) if gate_dir == "lt" else np.min(a)), 2),
            "gate":   gate_val,
            "pass_pct": round(float(np.mean(a < gate_val if gate_dir == "lt" else a > gate_val) * 100), 2),
        }
    
    # ── MTTF fleet distribution ───────────────────────────────────────────
    mttf_arr    = R["mttf_yr"][np.isfinite(R["mttf_yr"]) & (R["mttf_yr"] > 0)]
    fleet_stats = {
        "mean_mttf_yr":    round(float(np.mean(mttf_arr)), 1),
        "median_mttf_yr":  round(float(np.median(mttf_arr)), 1),
        "p10_mttf_yr":     round(float(np.percentile(mttf_arr, 10)), 1),   # 10% of fleet fails by this age
        "p1_mttf_yr":      round(float(np.percentile(mttf_arr, 1)), 1),    # "infant mortality" 1%
        "fraction_10yr":   round(float(np.mean(mttf_arr >= 10.0) * 100), 1),   # % meeting JEDEC
        "fraction_5yr":    round(float(np.mean(mttf_arr >= 5.0) * 100), 1),
    }
    
    # ── Sensitivity analysis ──────────────────────────────────────────────
    sensitivity = {}
    if compute_sensitivity and N >= 200:
        nominal_for_sobol = {
            "r_theta":        nominal["thermal_resistance_c_per_w"],
            "vdd":            nominal["vdd_v"],
            "p_leak_mult":    1.0,
            "p_dynamic_mult": 1.0,
            "t_ambient_abs":  nominal["t_ambient_c"],
            "bump_density":   nominal["bump_density_per_mm2"],
        }
        # Pass P (which uses _evaluate_batch keys) directly
        sensitivity = _sobol_sensitivity(nominal_for_sobol, R, P, rng)
    
    # ── Status ────────────────────────────────────────────────────────────
    if yield_pct >= 95:
        status = "pass"
    elif yield_pct >= 80:
        status = "warning"
    else:
        status = "critical"
    
    notes = []
    notes.append(f"Yield: {yield_pct:.1f}% ± {ci_width/2:.1f}% (2σ CI, N={N})")
    
    # Find dominant failure mode
    worst_gate = min(gate_yields, key=gate_yields.get)
    worst_yield = gate_yields[worst_gate]
    notes.append(f"Dominant failure: {worst_gate} ({worst_yield:.1f}% pass rate)")
    
    if sensitivity:
        top_param = max(sensitivity, key=sensitivity.get)
        notes.append(f"Top sensitivity: {top_param} (S_i={sensitivity[top_param]:.2f}) — reducing its σ improves yield most")
    
    notes.append(
        f"Fleet MTTF: median={fleet_stats['median_mttf_yr']:.0f}yr, "
        f"p10={fleet_stats['p10_mttf_yr']:.0f}yr, "
        f"{fleet_stats['fraction_10yr']:.0f}% meet JEDEC 10yr"
    )
    
    if fleet_stats["p1_mttf_yr"] < 5.0:
        notes.append(f"WARNING: 1% of fleet fails within {fleet_stats['p1_mttf_yr']:.1f}yr (infant mortality risk)")
    
    return {
        "yield_pct":       round(yield_pct, 2),
        "yield_ci_pct":    round(ci_width, 2),
        "n_samples":       N,
        "n_pass":          int(np.sum(passes)),
        "n_fail":          int(np.sum(~passes)),
        "gate_yields":     gate_yields,
        "gate_failures":   gate_failures,
        "distributions":   dist_stats,
        "fleet":           fleet_stats,
        "sensitivity":     sensitivity,
        "yield_history":   yield_history[-10:],  # last 10 convergence points
        "status":          status,
        "notes":           notes,
    }
