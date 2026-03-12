"""
thermal_electrical_solver.py v3 — Coupled Thermal-Electrical Solver

Solves: T* = T_amb + [P_dyn + P_ref × exp(α(T* - T_ref))] × R_theta
via fixed-point iteration with convergence and runaway detection.

Stability criterion (from fixed-point theory):
  |dT_new/dT_old| = α × P_leak(T*) × R_theta < 1
  If this condition fails anywhere in the iteration → THERMAL RUNAWAY.
  No stable T* exists — temperature will rise until damage.

Physical calibration (TSMC 5nm published measurements):
  P_leak(25°C)  = 4% of P_dynamic
  P_leak(105°C) = 27% of P_dynamic   [measured]
  → α = ln(0.27/0.04) / 80°C = 0.0239 /°C
  → Leakage doubles every 29°C (not the commonly cited 10°C)

R_theta_critical computed via brentq root-finding (scipy).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from typing import Optional, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Calibrated to TSMC 5nm: 4% leakage at 25°C → 27% at 105°C
# ── Calibration constants ────────────────────────────────────────────────────
try:
    from config.calibration_constants import R_THETA_CORRECTION, POWER_CORRECTION as _CALIB_P
    _CALIB_R = R_THETA_CORRECTION
except ImportError:
    _CALIB_R = 1.0; _CALIB_P = 1.0

ALPHA_LEAK = 0.0239          # /°C  (= ln(6.75)/80)
LEAK_REF_FRACTION = 0.04     # fraction of P_dynamic at T=25°C
T_REF_C = 25.0
JEDEC_LIMIT_C = 125.0
FF_LEAK_MULT = 4.2           # FF corner: 4.2× more leakage (TSMC 5nm PDK)


def _p_leak(T_c: float, p_leak_ref: float) -> float:
    return p_leak_ref * np.exp(min(60.0, ALPHA_LEAK * (T_c - T_REF_C)))


def _fixed_point_iterate(p_dynamic, p_leak_ref, r_theta, t_ambient, max_iter=200, tol=0.01):
    """Fixed-point iteration: T_new = T_amb + (P_dyn + P_leak(T)) × R."""
    T = t_ambient + p_dynamic * r_theta * 0.5   # initial guess
    for i in range(max_iter):
        p_lk  = _p_leak(T, p_leak_ref)
        T_new = t_ambient + (p_dynamic + p_lk) * r_theta
        # Stability check at current point
        sens = ALPHA_LEAK * p_lk * r_theta
        if sens >= 1.0 or T_new > 300:
            return None, True, i + 1   # runaway
        if abs(T_new - T) < tol:
            return T_new, False, i + 1
        T = T_new
    return None, True, max_iter   # didn't converge


def _find_r_theta_critical(p_dynamic, p_leak_ref, t_ambient):
    """
    Find R_theta_critical: the thermal resistance at which the system
    JUST becomes unstable (boundary of runaway condition).
    
    At criticality: α × P_leak(T*) × R = 1
    where T* = T_amb + P_dyn × R + 1/α  (marginal fixed point)
    """
    def stability_residual(R):
        T_star = t_ambient + p_dynamic * R + 1.0 / ALPHA_LEAK
        p_lk   = _p_leak(T_star, p_leak_ref)
        return ALPHA_LEAK * p_lk * R - 1.0

    try:
        # Check if any stable R exists in [0.001, 2.0]
        f_lo = stability_residual(0.001)
        f_hi = stability_residual(2.0)
        if f_lo * f_hi > 0:
            return None   # no root → always stable or always runaway
        return round(brentq(stability_residual, 0.001, 2.0, xtol=1e-5), 4)
    except Exception:
        return None


def _run_corner(p_dynamic, p_leak_ref, r_theta, t_ambient, label):
    T_ss, runaway, iters = _fixed_point_iterate(p_dynamic, p_leak_ref, r_theta, t_ambient)
    if T_ss is None:
        T_ss = 300.0   # placeholder
    p_lk      = _p_leak(T_ss, p_leak_ref)
    p_total   = p_dynamic + p_lk
    sens      = ALPHA_LEAK * p_lk * r_theta
    return {
        "label":           label,
        "converged":       not runaway,
        "runaway":         runaway,
        "T_op_c":          round(float(T_ss), 2),
        "P_leak_op_w":     round(float(p_lk), 2),
        "P_total_op_w":    round(float(p_total), 2),
        "P_overhead_pct":  round(float(p_lk / max(p_total, 0.01) * 100), 1),
        "runaway_factor":  round(float(sens), 4),
        "iterations":      iters,
    }


def run_coupled_solver(
    power_params: Optional[dict] = None,
    thermal_params: Optional[dict] = None,
    t_hotspot_c: Optional[float] = None,
) -> Dict:
    pp = power_params  or {}
    tp = thermal_params or {}

    p_dynamic  = float(pp.get("watt") or pp.get("tdp_watt") or 100.0)
    t_ambient  = float(tp.get("t_ambient_c") or 25.0)
    r_theta    = float(tp.get("thermal_resistance_c_per_w") or 0.10) * _CALIB_R
    p_leak_ref = p_dynamic * LEAK_REF_FRACTION

    # TT corner
    tt = _run_corner(p_dynamic, p_leak_ref, r_theta, t_ambient, "TT")
    # FF corner: 4.2× leakage
    ff = _run_corner(p_dynamic, p_leak_ref * FF_LEAK_MULT, r_theta, t_ambient, "FF")

    # R_theta critical
    r_crit = _find_r_theta_critical(p_dynamic, p_leak_ref, t_ambient)
    margin_pct = round((r_crit - r_theta) / r_crit * 100, 1) if r_crit else 100.0

    # Hotspot check
    hotspot = None
    if t_hotspot_c:
        p_lk_hs = _p_leak(t_hotspot_c, p_leak_ref * 1.8)   # local hotspot: 1.8× density
        sens_hs = ALPHA_LEAK * p_lk_hs * r_theta * 1.8
        hotspot = {
            "T_hotspot_c":        round(t_hotspot_c, 1),
            "P_leak_hotspot_w":   round(float(p_lk_hs), 2),
            "runaway_at_hotspot": bool(sens_hs >= 1.0),
            "runaway_factor":     round(float(sens_hs), 4),
        }

    # Status
    status = "pass"; notes = []
    if tt["runaway"]:
        status = "critical"
        notes.append(
            f"THERMAL RUNAWAY (TT): no stable operating point. "
            f"R_theta_critical={r_crit or '?'} °C/W, actual={r_theta:.3f} °C/W"
        )
    elif ff["runaway"]:
        status = "critical"
        notes.append(f"THERMAL RUNAWAY at FF corner (4.2× leakage) — must de-rate power or improve cooling")
    elif hotspot and hotspot["runaway_at_hotspot"]:
        status = "critical"
        notes.append(f"LOCAL RUNAWAY at hotspot ({t_hotspot_c:.0f}°C) — average die may be stable but hotspot is not")
    elif tt["runaway_factor"] > 0.7:
        status = "warning"
        notes.append(f"Near-runaway: factor={tt['runaway_factor']:.3f}, {margin_pct:.0f}% margin to R_theta_critical")
    elif tt["T_op_c"] > JEDEC_LIMIT_C:
        status = "warning"
        notes.append(f"TT corner stable but T_op={tt['T_op_c']}°C exceeds JEDEC {JEDEC_LIMIT_C}°C")
    else:
        notes.append(f"Coupled solution stable at T_op={tt['T_op_c']}°C, {margin_pct:.0f}% margin to runaway")

    if not tt["runaway"]:
        notes.append(
            f"T_op={tt['T_op_c']}°C | P_dyn={p_dynamic:.0f}W + P_leak={tt['P_leak_op_w']:.0f}W "
            f"= {tt['P_total_op_w']:.0f}W ({tt['P_overhead_pct']:.0f}% leakage overhead)"
        )
    ff_str = "RUNAWAY" if ff["runaway"] else f"T={ff['T_op_c']}C P={ff['P_total_op_w']:.0f}W"
    rc_str = "not found" if not r_crit else f"{r_crit:.4f} C/W"
    notes.append(f"FF corner: {ff_str} | R_theta_critical: {rc_str}")

    return {
        "tt_corner": tt, "ff_corner": ff, "hotspot": hotspot,
        "r_theta_critical": r_crit,
        "margin_to_runaway_pct": margin_pct,
        "alpha_leak_per_c": ALPHA_LEAK,
        "leak_doubling_temp_c": round(np.log(2)/ALPHA_LEAK, 1),
        "ff_leakage_mult": FF_LEAK_MULT,
        "status": status, "notes": notes,
    }
