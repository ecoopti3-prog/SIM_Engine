"""
L1 Thermal Simulator — RC Network (Lumped Parameter)
Model: T_j = T_amb + P * R_theta  (steady-state)
Transient: scipy.integrate.solve_ivp on C*dT/dt = P - (T-T_amb)/R_theta
Falsification: sweeps power to find exact T_junction = 125°C crossing
"""
from __future__ import annotations
import math
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Any, List
import sys, os
# Package root = sim_engine_v6/ (grandparent of this file's directory)
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from core.schemas import DomainSimResult, FalsificationBoundary, ParameterSweepPoint, SimStatus

T_JEDEC_MAX_C         = 125.0
HEAT_FLUX_COPPER_MAX  = 100.0   # W/cm²
HEAT_FLUX_DIAMOND_MAX = 500.0   # W/cm²
K_SILICON             = 1.5     # W/(cm·K)
K_DIAMOND             = 20.0
K_COPPER              = 4.0

def carnot_cop(t_cold_c: float, t_hot_c: float) -> float:
    tk, th = t_cold_c + 273.15, t_hot_c + 273.15
    return tk / (th - tk) if th > tk else float('inf')

def simulate_thermal(
    idea_id: str,
    power_w: Optional[float] = None,
    t_junction_c: Optional[float] = None,
    heat_flux_w_cm2: Optional[float] = None,
    thermal_resistance_c_per_w: Optional[float] = None,
    delta_t_c: Optional[float] = None,
    t_ambient_c: Optional[float] = None,
    cop_claimed: Optional[float] = None,
    material: Optional[str] = None,
    die_area_cm2: Optional[float] = None,
    t_j_limit: float = T_JEDEC_MAX_C,
) -> DomainSimResult:
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    sweep: List[ParameterSweepPoint] = []
    falsification: Optional[FalsificationBoundary] = None
    t_amb = t_ambient_c if t_ambient_c is not None else 25.0

    if not any([power_w, t_junction_c, heat_flux_w_cm2, thermal_resistance_c_per_w]):
        return DomainSimResult(
            idea_id=idea_id, domain="thermal", fidelity="L1_analytical",
            status="insufficient_data", score=4.0,
            warnings=["No thermal params — need power_w, R_theta, heat_flux, or T_junction"],
        )

    # ── Steady-state RC ────────────────────────────────────────────────────────
    t_j_sim: Optional[float] = None
    if power_w and thermal_resistance_c_per_w:
        t_j_sim = t_amb + power_w * thermal_resistance_c_per_w
        metrics["T_junction_c"]   = round(t_j_sim, 2)
        metrics["R_theta_c_per_w"] = thermal_resistance_c_per_w
        metrics["power_w"]         = power_w
        margin_c = t_j_limit - t_j_sim
        metrics["margin_to_jedec_c"]   = round(margin_c, 2)
        metrics["margin_to_jedec_pct"] = round(margin_c / t_j_limit * 100, 1)
        if t_j_sim > t_j_limit:
            warnings.append(f"T_junction {t_j_sim:.1f}°C EXCEEDS JEDEC {t_j_limit}°C by {-margin_c:.1f}°C")
        elif margin_c / t_j_limit < 0.10:
            warnings.append(f"T_junction within 10% of JEDEC limit — MARGINAL ({margin_c:.1f}°C headroom)")
        # Power sweep for falsification
        r = thermal_resistance_c_per_w
        p_break = (t_j_limit - t_amb) / r
        margin_pct = (p_break - power_w) / p_break * 100
        for frac in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]:
            p = power_w * frac
            tj = t_amb + p * r
            sweep.append(ParameterSweepPoint(
                param_name="power_w", param_value=round(p, 1),
                metric_name="T_junction_c", metric_value=round(tj, 2),
                status="pass" if tj < t_j_limit else "fail",
            ))
        falsification = FalsificationBoundary(
            domain="thermal", breaking_param="power_w",
            nominal_value=round(power_w, 1), breaking_value=round(p_break, 1),
            safety_margin_pct=round(margin_pct, 1), units="W",
            equation_used=f"T_j = {t_amb} + P × {r:.4f}  (limit={t_j_limit}°C)",
            fix_vector=(
                f"Need R_theta ≤ {(t_j_limit - t_amb)/power_w:.3f} °C/W. "
                "Options: two-phase vapor chamber, microfluidic cooling, diamond TIM."
            ) if margin_pct < 0 else f"{margin_pct:.1f}% power headroom before thermal death."
        )
    elif t_junction_c:
        t_j_sim = t_junction_c
        metrics["T_junction_c"] = t_j_sim
        margin_c = t_j_limit - t_j_sim
        metrics["margin_to_jedec_c"] = round(margin_c, 2)
        if t_j_sim > t_j_limit:
            warnings.append(f"Stated T_junction {t_j_sim:.1f}°C EXCEEDS JEDEC {t_j_limit}°C")

    # ── Heat flux check ────────────────────────────────────────────────────────
    if heat_flux_w_cm2:
        mat = (material or "copper").lower()
        lim = HEAT_FLUX_DIAMOND_MAX if "diamond" in mat else HEAT_FLUX_COPPER_MAX
        flux_margin_pct = (lim - heat_flux_w_cm2) / lim * 100
        metrics["heat_flux_w_cm2"] = heat_flux_w_cm2
        metrics["heat_flux_limit"]  = lim
        metrics["flux_margin_pct"]  = round(flux_margin_pct, 1)
        if heat_flux_w_cm2 > lim:
            warnings.append(f"Heat flux {heat_flux_w_cm2} W/cm² EXCEEDS {mat} limit {lim} W/cm²")
        if not falsification:
            falsification = FalsificationBoundary(
                domain="thermal", breaking_param="heat_flux_w_cm2",
                nominal_value=heat_flux_w_cm2, breaking_value=lim,
                safety_margin_pct=round(flux_margin_pct, 1), units="W/cm²",
                equation_used=f"practical_limit({mat}) = {lim} W/cm²",
                fix_vector=(
                    "Reduce power density or increase die area. "
                    + ("Consider diamond substrate (500 W/cm² limit)." if "copper" in mat else "")
                ),
            )

    # ── Carnot COP check ───────────────────────────────────────────────────────
    if cop_claimed is not None:
        dt = delta_t_c or 30.0
        cop_max = carnot_cop(t_amb, t_amb + dt)
        metrics["COP_claimed"] = cop_claimed
        metrics["COP_carnot"]  = round(cop_max, 3)
        metrics["COP_efficiency_pct"] = round(cop_claimed / cop_max * 100, 1)
        if cop_claimed > cop_max * 1.02:
            warnings.append(
                f"COP_claimed={cop_claimed:.2f} VIOLATES Carnot max={cop_max:.2f} — PHYSICALLY IMPOSSIBLE"
            )

    # ── Transient simulation (if R_theta and power known) ─────────────────────
    if power_w and thermal_resistance_c_per_w:
        r, p = thermal_resistance_c_per_w, power_w
        c_thermal = 50.0  # J/°C — typical chiplet assumption
        def ode(t, T): return [(p - (T[0] - t_amb) / r) / c_thermal]
        sol = solve_ivp(ode, [0, 0.5], [t_amb], max_step=0.01, dense_output=True)
        t_final = float(sol.y[0, -1])
        metrics["T_steady_state_c"]  = round(t_final, 2)
        metrics["thermal_time_constant_s"] = round(r * c_thermal, 3)
        # Store transient trace at key time points
        t_trace = np.linspace(0, 0.5, 20)
        T_trace = sol.sol(t_trace)[0]
        sweep.extend([
            ParameterSweepPoint(
                param_name="time_s", param_value=round(float(tt), 3),
                metric_name="T_junction_transient_c", metric_value=round(float(Tv), 2),
                status="pass" if Tv < t_j_limit else "fail",
            ) for tt, Tv in zip(t_trace[1::4], T_trace[1::4])
        ])

    # ── Score & status ─────────────────────────────────────────────────────────
    fatal    = any("EXCEED" in w or "VIOLAT" in w or "IMPOSSIBLE" in w for w in warnings)
    marginal = any("MARGINAL" in w or "10%" in w for w in warnings)
    carnot_fail = any("Carnot" in w and "VIOLATES" in w for w in warnings)

    if carnot_fail:
        return DomainSimResult(
            idea_id=idea_id, domain="thermal", fidelity="L1_analytical",
            status="fail", score=0.0, metrics=metrics, sweep_points=sweep,
            falsification=falsification, warnings=warnings,
            solver_notes="Carnot violation — idea is physically impossible",
        )
    if fatal:
        score = 1.5
        status_f: SimStatus = "fail"
    elif marginal:
        score = 5.5
        status_f = "marginal"
    elif t_j_sim is not None:
        m = metrics.get("margin_to_jedec_pct", 50.0)
        score = min(10.0, 5.0 + m * 0.1)
        status_f = "pass"
    else:
        score = 4.0
        status_f = "insufficient_data"

    return DomainSimResult(
        idea_id=idea_id, domain="thermal", fidelity="L1_analytical",
        status=status_f, score=round(score, 2),
        metrics=metrics, sweep_points=sweep, falsification=falsification,
        warnings=warnings, solver_notes="RC steady-state + transient (scipy.solve_ivp)",
    )
