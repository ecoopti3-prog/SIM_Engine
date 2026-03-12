"""
pvt_corners.py — Process-Voltage-Temperature (PVT) Corner Analysis

Reality: every silicon wafer is different. TSMC/Samsung/Intel test every design
across 6 corners at tape-out. If any corner fails → redesign.

The 3 key corners for this engine:
  TT (Typical-Typical):   nominal process, VDD nominal, T = 25°C
  FF (Fast-Fast):         NMOS/PMOS both fast → leakage 3-5x, power spikes
  SS (Slow-Slow):         NMOS/PMOS both slow → timing at risk under voltage droop

The SS corner under voltage droop is the killer:
  SS chip already has slower transistors → then PDN droop reduces VDD further
  → timing violations → bit errors in compute → silent data corruption.

Physical basis:
  t_pd ∝ 1 / (k*(Vgs - Vt)²)    [CMOS propagation delay, Shockley MOSFET]
  Vt(T) decreases ~2mV/°C        [threshold voltage temperature coefficient]
  I_leak(T) = I0 * exp(ΔΦ/nVth) * exp(-Vt/(nVth))   [subthreshold leakage]

  FF corner: Vt LOW → I_leak HIGH by 3-5x, but timing fast
  SS corner: Vt HIGH → I_leak low, but t_pd HIGH → setup violations at droop
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ── PVT Corner Parameters ─────────────────────────────────────────────────────
# Based on published data from TSMC 5nm/3nm PDK corner definitions

CORNERS = {
    "TT": {   # Typical-Typical
        "vt_offset_mv":    0.0,       # threshold voltage offset from nominal
        "tox_scale":       1.0,       # gate oxide thickness scale (1 = nominal)
        "mobility_scale":  1.0,       # carrier mobility scale
        "leakage_mult":    1.0,       # leakage multiplier vs TT
        "delay_mult":      1.0,       # timing delay multiplier vs TT
        "description": "Nominal process — design target",
    },
    "FF": {   # Fast-Fast (worst-case power)
        "vt_offset_mv":   -30.0,      # lower Vt → more leakage, faster switching
        "tox_scale":       0.97,      # thinner oxide → more Cox → more I_leak
        "mobility_scale":  1.08,      # higher mobility → more leakage
        "leakage_mult":    4.2,       # leakage 4.2× higher (published TSMC 5nm data)
        "delay_mult":      0.88,      # 12% faster timing
        "description": "Fast-Fast — worst-case POWER (leakage dominates)",
    },
    "SS": {   # Slow-Slow (worst-case timing)
        "vt_offset_mv":   +30.0,      # higher Vt → less leakage, slower switching
        "tox_scale":       1.03,      # thicker oxide → less Cox
        "mobility_scale":  0.93,      # lower mobility
        "leakage_mult":    0.22,      # leakage 5× lower
        "delay_mult":      1.15,      # 15% slower timing → risk of setup violation
        "description": "Slow-Slow — worst-case TIMING (setup violations under droop)",
    },
}

# Temperature corners (often combined with PVT)
TEMP_CORNERS = {
    "T_cold":  0.0,    # cold start
    "T_typ":  25.0,    # room temperature
    "T_hot": 105.0,    # operating junction temperature (near JEDEC limit)
}

# Timing margin parameters
CLOCK_PERIOD_NS_BASE = 0.5     # 2GHz clock (conservative AI chip)
SETUP_MARGIN_PS      = 50.0    # required setup time margin [ps]


def _compute_timing_slack(
    corner: dict,
    vdd_nominal: float,
    vdd_actual: float,      # reduced by PDN droop
    t_junction_c: float,
    clock_period_ns: float = CLOCK_PERIOD_NS_BASE,
) -> Dict:
    """
    Compute timing slack under PVT + voltage droop.

    Uses simplified CMOS delay model:
    t_pd ≈ t_pd_nom × delay_mult × (Vdd_nom / Vdd_actual)^α × (1 - β*(T-T_nom))
    where α ≈ 1.3 (voltage sensitivity), β ≈ 0.003/°C (temperature sensitivity)
    """
    alpha_v = 1.3    # voltage exponent for delay
    beta_t  = 0.003  # delay temperature coefficient [1/°C]

    t_nom = CLOCK_PERIOD_NS_BASE * 0.6   # logic depth fills ~60% of clock period

    # Delay scaling
    v_scale = (vdd_nominal / max(vdd_actual, 0.5)) ** alpha_v
    t_scale = 1.0 + beta_t * (t_junction_c - TEMP_CORNERS["T_typ"])
    t_pd    = t_nom * corner["delay_mult"] * v_scale * t_scale

    slack_ps = (clock_period_ns - t_pd) * 1000 - SETUP_MARGIN_PS
    return {
        "t_pd_ps":     round(t_pd * 1000, 1),
        "slack_ps":    round(slack_ps, 1),
        "timing_ok":   bool(slack_ps > 0),
        "v_scale":     round(v_scale, 3),
        "t_scale":     round(t_scale, 3),
    }


def run_pvt_analysis(
    power_params: Optional[dict] = None,
    thermal_params: Optional[dict] = None,
    pdn_params: Optional[dict] = None,
    node_nm: float = 5.0,
) -> Dict:
    """
    Run PVT corner analysis across TT / FF / SS corners.

    Returns a dict with per-corner results + worst-case identification.
    """
    pp  = power_params  or {}
    tp  = thermal_params or {}
    pdn = pdn_params    or {}

    vdd_nom   = float(pp.get("voltage_v")  or pdn.get("vdd_v") or 0.85)
    p_dynamic = float(pp.get("watt") or pp.get("tdp_watt") or 100.0)
    t_junction = float(tp.get("t_junction_c") or
                       (tp.get("t_ambient_c") or 25.0) + 60.0)

    # PDN droop at worst case: SS corner has more delay, droop hits harder
    ir_drop_mv = float(pdn.get("ir_drop_mv") or 0)
    vdd_drooped = vdd_nom - ir_drop_mv * 1e-3

    # Leakage reference at TT corner, T_junction
    # I_leak(TT, T) = I_ref * exp(0.02*(T - 25))  [calibrated to 5%->25% TDP range]
    p_leak_ref_tt = p_dynamic * 0.05
    exp_arg       = min(10.0, 0.02 * (t_junction - TEMP_CORNERS["T_typ"]))
    p_leak_tt     = p_leak_ref_tt * np.exp(exp_arg)

    corner_results = {}
    for corner_name, corner in CORNERS.items():
        # Leakage scales with corner multiplier
        p_leak = p_leak_tt * corner["leakage_mult"]
        p_total = p_dynamic + p_leak

        # Additional self-heating from leakage changes T slightly
        r_theta = float(tp.get("thermal_resistance_c_per_w") or 0.15)
        delta_t_from_leak = (p_leak - p_leak_tt) * r_theta
        t_actual = t_junction + delta_t_from_leak

        # Timing analysis
        timing = _compute_timing_slack(
            corner, vdd_nom, vdd_drooped, t_actual
        )

        # Power density
        pd = pp.get("power_density_w_cm2")
        if pd:
            pd_corner = float(pd) * (p_total / (p_dynamic + p_leak_tt))
        else:
            pd_corner = None

        corner_results[corner_name] = {
            "p_dynamic_w":       round(p_dynamic, 2),
            "p_leakage_w":       round(p_leak, 2),
            "p_total_w":         round(p_total, 2),
            "leakage_fraction":  round(p_leak / p_total, 3),
            "t_junction_c":      round(t_actual, 1),
            "power_density_w_cm2": round(pd_corner, 1) if pd_corner else None,
            "timing":            timing,
            "description":       corner["description"],
            "leakage_mult":      corner["leakage_mult"],
            "delay_mult":        corner["delay_mult"],
        }

    # Identify worst cases
    max_power_corner  = max(corner_results, key=lambda k: corner_results[k]["p_total_w"])
    max_timing_risk   = min(corner_results, key=lambda k: corner_results[k]["timing"]["slack_ps"])
    min_slack_ps      = corner_results[max_timing_risk]["timing"]["slack_ps"]
    any_timing_fail   = any(not r["timing"]["timing_ok"] for r in corner_results.values())

    ff_p   = corner_results["FF"]["p_total_w"]
    tt_p   = corner_results["TT"]["p_total_w"]
    power_overhead_ff_pct = round((ff_p - tt_p) / tt_p * 100, 1)

    # Status
    status = "pass"
    notes  = []

    if any_timing_fail:
        status = "critical"
        notes.append(f"TIMING VIOLATION at SS corner: slack={min_slack_ps:.0f}ps (under PDN droop {ir_drop_mv:.0f}mV)")
    elif min_slack_ps < 30:
        status = "warning"
        notes.append(f"Timing margin tight at SS+droop: only {min_slack_ps:.0f}ps slack")
    if power_overhead_ff_pct > 30:
        if status == "pass":
            status = "warning"
        notes.append(f"FF corner adds {power_overhead_ff_pct:.0f}% power overhead — thermal budget must accommodate")

    notes.append(f"TT: {tt_p:.1f}W total | FF: {ff_p:.1f}W (+{power_overhead_ff_pct:.0f}%) | SS: {corner_results['SS']['p_total_w']:.1f}W")
    notes.append(f"Worst timing: {max_timing_risk} corner, slack={min_slack_ps:.0f}ps at Vdd={vdd_drooped:.3f}V")

    return {
        "corners":                corner_results,
        "worst_power_corner":     max_power_corner,
        "worst_timing_corner":    max_timing_risk,
        "min_timing_slack_ps":    round(min_slack_ps, 1),
        "any_timing_failure":     any_timing_fail,
        "power_overhead_ff_pct":  power_overhead_ff_pct,
        "vdd_drooped_v":          round(vdd_drooped, 4),
        "status":                 status,
        "notes":                  notes,
    }
