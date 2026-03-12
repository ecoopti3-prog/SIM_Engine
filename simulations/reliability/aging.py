"""
aging.py — Reliability & Aging: NBTI + HCI + Electromigration

JEDEC target: MTTF ≥ 10 years at operating conditions.

NBTI: PMOS threshold voltage shift → timing violations after months/years.
HCI:  NMOS degradation at high Vdd → MTTF from Black's law variant.
EM:   Cu wire atoms migrate → open circuit. Black's equation.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

KB_EV = 8.617e-5
JEDEC_LIFETIME_YEARS = 10.0

NBTI_A  = 0.0045; NBTI_n = 0.20; NBTI_Ea = 0.50; NBTI_MAX_VT_MV = 25.0
HCI_A   = 1e15;   HCI_b  = 3.0;  HCI_Ea  = -0.1
EM_A    = 1.48e11; EM_n  = 2.0;  EM_Ea   = 0.7;  EM_J_REF = 1e6
LEAK_DOUBLING_TEMP = 8.0   # TSMC 5nm: leakage doubles every 8°C


def _nbti(t_years, t_jc, duty_cycle=0.5):
    T_k = t_jc + 273.15
    t_s = t_years * 3.156e7
    # Empirical NBTI model calibrated to TSMC 5nm published measurements
    # ΔVt = 20mV × (T/378K)^3 × (duty)^0.3 × (t_yr/10)^0.18
    T_norm = T_k / 378.0      # normalize to 105°C reference
    t_norm = t_years / 10.0   # normalize to 10yr reference
    dVt    = 0.020 * (T_norm**3) * (duty_cycle**0.3) * (t_norm**0.18)
    dVt_mv = abs(dVt)*1000
    dtpd   = (dVt_mv*1e-3/0.30)*300
    return {"delta_vt_mv": round(dVt_mv,3), "delta_t_pd_ps": round(dtpd,1),
            "exceeds_limit": dVt_mv > NBTI_MAX_VT_MV, "limit_mv": NBTI_MAX_VT_MV}


def _hci(t_jc, vdd_v):
    T_k = t_jc + 273.15
    mttf_s = HCI_A * np.exp(HCI_b/np.sqrt(max(vdd_v,0.1))) * np.exp(-HCI_Ea/(KB_EV*T_k))
    y = mttf_s / 3.156e7
    return {"mttf_years": round(y,1), "mttf_ok": y >= JEDEC_LIFETIME_YEARS}


def _em(i_a, wire_area_um2, t_jc):
    T_k    = t_jc + 273.15
    area   = wire_area_um2 * 1e-8
    J      = i_a / max(area, 1e-12)
    jr     = J / EM_J_REF
    if jr > 1e6:
        y = 0.0
    else:
        mttf_s = EM_A / (J**EM_n) * np.exp(EM_Ea/(KB_EV*T_k))
        y = mttf_s / 3.156e7
    return {"J_a_cm2": round(J,2), "j_ratio": round(jr,2),
            "mttf_years": round(y,1), "mttf_ok": y >= JEDEC_LIFETIME_YEARS,
            "margin_pct": round((y-JEDEC_LIFETIME_YEARS)/JEDEC_LIFETIME_YEARS*100,1)}


def run_aging_analysis(power_params=None, thermal_params=None, pdn_params=None,
                       mission_years=JEDEC_LIFETIME_YEARS, duty_cycle=0.60):
    pp = power_params or {}; tp = thermal_params or {}; pdn = pdn_params or {}
    vdd_v = float(pp.get("voltage_v") or pdn.get("vdd_v") or 0.85)
    t_jc  = float(tp.get("t_junction_c") or (tp.get("t_ambient_c") or 25.0)+60.0)
    i_tot = float(pp.get("current_a") or pdn.get("current_a") or 200.0)
    bump_d = float(pdn.get("bump_density_per_mm2") or 2000.0)

    die_area_mm2 = 500.0
    if pp.get("power_density_w_cm2") and pp.get("watt"):
        die_area_mm2 = float(pp["watt"]) / float(pp["power_density_w_cm2"]) * 100
    n_wires    = max(1, int(bump_d * die_area_mm2 * 4))
    i_per_wire = i_tot / n_wires

    nbti = _nbti(mission_years, t_jc, duty_cycle)
    hci  = _hci(t_jc, vdd_v)
    em   = _em(i_per_wire, 0.08, t_jc)

    min_mttf = min(v for v in [hci["mttf_years"], em["mttf_years"]] if v > 0) if True else 0.0
    failures = []; warnings = []; notes = []

    if nbti["exceeds_limit"]:
        failures.append(f"NBTI: ΔVt={nbti['delta_vt_mv']:.1f}mV > {NBTI_MAX_VT_MV}mV — timing degrades to failure in {mission_years:.0f}yr")
    if not hci["mttf_ok"]:
        failures.append(f"HCI: MTTF={hci['mttf_years']:.1f}yr < {JEDEC_LIFETIME_YEARS:.0f}yr JEDEC minimum")
    if not em["mttf_ok"]:
        failures.append(f"EM: MTTF={em['mttf_years']:.1f}yr — J={em['J_a_cm2']:.1e} A/cm² exceeds limit")
    if nbti["delta_vt_mv"] > NBTI_MAX_VT_MV * 0.7:
        warnings.append(f"NBTI marginal: {nbti['delta_vt_mv']:.1f}mV ({nbti['delta_vt_mv']/NBTI_MAX_VT_MV*100:.0f}% of budget)")

    status = "fail" if failures else ("warning" if warnings else "pass")
    notes  = failures + warnings
    notes.append(f"NBTI ΔVt={nbti['delta_vt_mv']:.1f}mV (+{nbti['delta_t_pd_ps']:.0f}ps) | HCI MTTF={hci['mttf_years']:.0f}yr | EM MTTF={em['mttf_years']:.0f}yr")
    notes.append(f"Conditions: T_j={t_jc:.0f}°C, Vdd={vdd_v}V, {duty_cycle:.0%} duty, {mission_years:.0f}yr mission")

    return {"nbti": nbti, "hci": hci, "em": em, "min_mttf_years": min_mttf,
            "mission_years": mission_years, "duty_cycle": duty_cycle,
            "t_junction_c": t_jc, "status": status, "notes": notes}
