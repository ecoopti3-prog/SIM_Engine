"""
rc_network.py — Multi-node transient thermal simulation.

Solves 3-node thermal RC ODE system:
  Node 0: die (silicon junction)
  Node 1: IHS (integrated heat spreader, Cu)
  Node 2: heatsink

Key physics beyond the RD Engine's static gate:
  - Transient T(t): rise time, peak overshoot
  - Leakage-thermal feedback loop (Q10 model)
  - Thermal runaway detection: dP_leak/dT * R_theta >= 1 → unstable
  - Dominant resistance identification
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.schemas import ThermalSimResult

DEFAULT_NET = {
    "R_die_tim1":  0.04, "R_tim1_ihs":  0.02,
    "R_ihs_tim2":  0.06, "R_tim2_hs":   0.03, "R_hs_amb": 0.10,
    "C_die": 0.08, "C_ihs": 12.0, "C_hs": 80.0,
    "I_leak_ref_a": 0.5, "T_ref_c": 25.0, "Q10": 2.0,
}
JEDEC = {"standard": 125.0, "automotive": 150.0, "default": 125.0}



# ── Calibration constants (auto-updated by fit_thermal.py) ───────────────────
try:
    from config.calibration_constants import R_THETA_CORRECTION, POWER_CORRECTION
    _CALIB_R = R_THETA_CORRECTION
    _CALIB_P = POWER_CORRECTION
except ImportError:
    _CALIB_R = 1.0   # no calibration yet → use datasheet values
    _CALIB_P = 1.0

def _build_ode(net, p_dynamic_w, t_ambient, vdd_v):
    R_die_ihs = net["R_die_tim1"] + net["R_tim1_ihs"]
    R_ihs_hs  = net["R_ihs_tim2"] + net["R_tim2_hs"]
    C_die, C_ihs, C_hs = net["C_die"], net["C_ihs"], net["C_hs"]
    I_ref, T_ref, Q10  = net["I_leak_ref_a"], net["T_ref_c"], net["Q10"]

    def ode(t, T):
        T_die, T_ihs, T_hs = T
        p_leak   = I_ref * vdd_v * (Q10 ** ((T_die - T_ref) / 10.0))
        p_total  = p_dynamic_w + p_leak
        q_1 = (T_die - T_ihs) / R_die_ihs
        q_2 = (T_ihs - T_hs)  / R_ihs_hs
        q_3 = (T_hs  - t_ambient) / net["R_hs_amb"]
        return [(p_total - q_1)/C_die, (q_1 - q_2)/C_ihs, (q_2 - q_3)/C_hs]
    return ode


def run_thermal_sim(
    power_w: float,
    thermal_params: Optional[dict] = None,
    t_ambient_c: float = 25.0,
    vdd_v: float = 0.85,
    t_sim_s: float = 2.0,
    jedec_grade: str = "standard",
) -> ThermalSimResult:
    result = ThermalSimResult()
    net = DEFAULT_NET.copy()

    if thermal_params:
        if thermal_params.get("t_ambient_c"):
            t_ambient_c = thermal_params["t_ambient_c"]
        if thermal_params.get("thermal_resistance_c_per_w"):
            r_total = thermal_params["thermal_resistance_c_per_w"]
            r_total *= _CALIB_R  # apply physical calibration correction
            r_def = sum(net[k] for k in ["R_die_tim1","R_tim1_ihs","R_ihs_tim2","R_tim2_hs","R_hs_amb"])
            scale = r_total / r_def
            for k in ["R_die_tim1","R_tim1_ihs","R_ihs_tim2","R_tim2_hs","R_hs_amb"]:
                net[k] *= scale

    jedec_limit = JEDEC.get(jedec_grade, 125.0)
    T0     = [t_ambient_c]*3
    t_eval = np.linspace(0, t_sim_s, 2000)

    sol = solve_ivp(_build_ode(net, power_w, t_ambient_c, vdd_v),
                    (0, t_sim_s), T0, method="Radau",
                    t_eval=t_eval, rtol=1e-5, atol=1e-7)

    T_die = sol.y[0]
    t_vec = sol.t
    t_ss   = float(T_die[-1])
    t_peak = float(T_die.max())

    # Leakage at SS
    p_leak_ss = net["I_leak_ref_a"] * vdd_v * (net["Q10"] ** ((t_ss - net["T_ref_c"]) / 10.0))
    result.p_dynamic_w         = round(power_w, 3)
    result.p_leakage_w         = round(p_leak_ss, 3)
    result.t_junction_ss_c     = round(t_ss, 2)
    result.t_peak_transient_c  = round(t_peak, 2)
    result.thermal_margin_pct  = round(((jedec_limit - t_ss) / jedec_limit) * 100, 1)

    # 90% rise time
    delta    = t_ss - t_ambient_c
    t90      = t_ambient_c + 0.90 * delta
    idx_90   = np.searchsorted(T_die, t90)
    if idx_90 < len(t_vec):
        result.t_rise_90pct_ms = round(float(t_vec[idx_90]) * 1000, 1)

    # Dominant bottleneck
    segs = {
        "die→IHS":          net["R_die_tim1"] + net["R_tim1_ihs"],
        "IHS→heatsink":     net["R_ihs_tim2"] + net["R_tim2_hs"],
        "heatsink→ambient": net["R_hs_amb"],
    }
    result.dominant_resistance = max(segs, key=segs.get)

    # Thermal runaway: dP_leak/dT * R_total >= 1 → unstable
    dP_dT = p_leak_ss * np.log(net["Q10"]) / 10.0
    r_total = sum(net[k] for k in ["R_die_tim1","R_tim1_ihs","R_ihs_tim2","R_tim2_hs","R_hs_amb"])
    runaway_factor = dP_dT * r_total
    result.thermal_runaway_risk = bool(runaway_factor >= 1.0)

    # Downsample time series
    step = max(1, len(t_vec) // 200)
    result.time_s               = [round(float(x), 5) for x in t_vec[::step]]
    result.t_junction_transient = [round(float(x), 3) for x in T_die[::step]]

    # Status
    if result.thermal_runaway_risk:
        result.status = "critical"
        result.notes.append(f"THERMAL RUNAWAY: feedback factor={runaway_factor:.2f}")
    elif t_ss > jedec_limit:
        result.status = "fail"
        result.notes.append(f"T_j={t_ss:.1f}°C > JEDEC {jedec_limit}°C")
    elif result.thermal_margin_pct < 10:
        result.status = "critical"
        result.notes.append(f"Only {result.thermal_margin_pct:.1f}% margin to JEDEC")
    elif result.thermal_margin_pct < 25:
        result.status = "warning"
        result.notes.append(f"{result.thermal_margin_pct:.1f}% JEDEC margin — tight budget")
    else:
        result.status = "pass"
        result.notes.append(f"T_j={t_ss:.1f}°C, {result.thermal_margin_pct:.1f}% margin")

    result.notes.append(f"Leakage: {p_leak_ss:.2f}W ({p_leak_ss/(power_w+p_leak_ss)*100:.1f}% of total)")
    result.notes.append(f"Bottleneck: {result.dominant_resistance}")
    if result.t_rise_90pct_ms:
        result.notes.append(f"90% thermal rise time: {result.t_rise_90pct_ms:.0f}ms")
    return result
