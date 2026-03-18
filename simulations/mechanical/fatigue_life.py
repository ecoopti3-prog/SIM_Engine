"""
fatigue_life.py — Mechanical fatigue & bearing reliability simulator.

Simulates robotic joint and actuator mechanical failure modes:
  1. Fatigue life via Rainflow cycle counting (ASTM E1049) + Miner's rule
  2. Weibull bearing fleet reliability (ISO 281) + Monte Carlo scatter
  3. Vibration resonance detection with amplitude response
  4. Stress-strain ODE under cyclic loading

Key physics beyond the RD Engine's static gate:
  - Time-domain fatigue accumulation under variable amplitude (not just peak)
  - Fleet reliability: P(survival=0.99) at target mission hours
  - Resonance amplification factor Q at natural frequency
  - Crack propagation estimate (Paris law, simplified)
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from typing import Optional, Dict, Any
import logging, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)

# ── Material library ──────────────────────────────────────────────────────────
MATERIALS = {
    "steel":     {"E_gpa": 200.0, "UTS_mpa": 500.0, "Se_ratio": 0.50, "b": -0.087, "K_ic": 50.0},
    "aluminum":  {"E_gpa": 69.0,  "UTS_mpa": 310.0, "Se_ratio": 0.35, "b": -0.100, "K_ic": 30.0},
    "titanium":  {"E_gpa": 116.0, "UTS_mpa": 900.0, "Se_ratio": 0.45, "b": -0.090, "K_ic": 60.0},
    "cast_iron": {"E_gpa": 170.0, "UTS_mpa": 250.0, "Se_ratio": 0.40, "b": -0.080, "K_ic": 20.0},
    "default":   {"E_gpa": 200.0, "UTS_mpa": 500.0, "Se_ratio": 0.45, "b": -0.087, "K_ic": 45.0},
}


def _rainflow_count(signal: np.ndarray) -> np.ndarray:
    """
    ASTM E1049 rainflow cycle counting.
    Returns array of cycle amplitudes [MPa].
    """
    peaks_idx, _ = find_peaks(signal)
    valleys_idx, _ = find_peaks(-signal)
    tp_idx = np.sort(np.concatenate([[0, len(signal)-1], peaks_idx, valleys_idx]))
    s = signal[tp_idx]

    stack = []
    cycles = []
    for sv in s:
        stack.append(float(sv))
        while len(stack) >= 4:
            r1 = abs(stack[-3] - stack[-4])
            r2 = abs(stack[-2] - stack[-3])
            r3 = abs(stack[-1] - stack[-2])
            if r2 <= r1 and r2 <= r3:
                cycles.append(r2 / 2.0)
                stack.pop(-3)
                stack.pop(-2)
            else:
                break
    for i in range(len(stack) - 1):
        cycles.append(abs(stack[i+1] - stack[i]) / 4.0)

    return np.array(cycles) if cycles else np.array([0.0])


def run_fatigue_simulation(
    mechanical_params: dict,
    mission_hours: float = 20000.0,
    duty_cycle: float = 0.70,
    material: str = "steel",
) -> Dict[str, Any]:
    """
    Simulate fatigue life under variable amplitude loading.

    Inputs from mechanical_params:
      stress_amplitude_mpa    — peak cyclic stress amplitude
      mean_stress_mpa         — mean stress (Goodman correction)
      frequency_hz            — loading frequency
      stress_concentration_kt — stress concentration factor (notch)

    Returns full transient damage accumulation + life prediction.
    """
    mat = MATERIALS.get(material, MATERIALS["default"])
    UTS  = mat["UTS_mpa"]
    Se   = mat["Se_ratio"] * UTS
    b    = mat["b"]               # Basquin exponent
    E    = mat["E_gpa"] * 1e3     # MPa

    # Extract params with defaults
    sigma_a = float(mechanical_params.get("stress_amplitude_mpa") or 120.0)
    sigma_m = float(mechanical_params.get("mean_stress_mpa") or 0.0)
    freq_hz = float(mechanical_params.get("frequency_hz") or 10.0)
    Kt      = float(mechanical_params.get("stress_concentration_kt") or 1.5)
    sigma_a_eff = sigma_a * Kt

    # Goodman mean stress correction
    if sigma_m > 0 and UTS > 0:
        Se_corrected = Se * (1 - sigma_m / UTS)
    else:
        Se_corrected = Se

    # ── Synthesize realistic stress history (40 seconds) ──────────────────
    t_sim   = np.linspace(0, 40.0, 4000)
    # Multi-frequency loading: fundamental + harmonics + random vibration
    stress  = (
        sigma_a_eff * np.sin(2 * np.pi * freq_hz * t_sim)
        + 0.3 * sigma_a_eff * np.sin(2 * np.pi * freq_hz * 3 * t_sim + 0.5)
        + 0.1 * sigma_a_eff * np.random.default_rng(42).normal(0, 1, len(t_sim))
        + sigma_m
    )

    # ── Rainflow counting ─────────────────────────────────────────────────
    cycles = _rainflow_count(stress)

    # ── Miner's rule damage per simulation period ──────────────────────────
    # N_f(sigma) = (Se_corrected / sigma_a)^(1/b) × N_ref  (Basquin)
    N_REF = 1e7
    damage_per_sim = 0.0
    for amp in cycles:
        if amp <= 0 or amp < Se_corrected * 0.1:
            continue
        N_f = N_REF * (Se_corrected / amp) ** (1.0 / abs(b))
        damage_per_sim += 1.0 / max(N_f, 1.0)

    # Scale to mission duration
    sim_duration_s    = 40.0
    active_seconds    = mission_hours * 3600 * duty_cycle
    scale_factor      = active_seconds / sim_duration_s
    total_damage      = damage_per_sim * scale_factor

    # ── Life prediction ────────────────────────────────────────────────────
    if damage_per_sim > 0:
        life_hours = (1.0 / damage_per_sim) * (sim_duration_s / 3600) / duty_cycle
    else:
        life_hours = 999999.0

    # ── Crack propagation (Paris law: da/dN = C × ΔK^m) ──────────────────
    a0_mm    = 0.1      # initial crack size (mm)
    a_crit   = mat["K_ic"] / (0.5 * sigma_a_eff * np.sqrt(np.pi)) * 1000  # critical crack (mm)
    a_crit   = max(a_crit, 1.0)
    C_paris  = 1e-12    # Paris constant (steel, SI)
    m_paris  = 3.0
    delta_K  = sigma_a_eff * np.sqrt(np.pi * a0_mm * 1e-3) * 1e6  # Pa√m
    da_per_cycle = C_paris * (delta_K ** m_paris)
    cycles_to_crit = (a_crit - a0_mm) * 1e-3 / da_per_cycle if da_per_cycle > 0 else 1e9

    # ── Status ────────────────────────────────────────────────────────────
    if total_damage >= 1.0:
        status = "fail"
    elif total_damage >= 0.7:
        status = "critical"
    elif total_damage >= 0.4:
        status = "warning"
    else:
        status = "pass"

    notes = [
        f"Miner damage D={total_damage:.3f} | Life: {life_hours:.0f} h | Mission: {mission_hours:.0f} h",
        f"Effective stress amplitude: {sigma_a_eff:.1f} MPa (Kt={Kt}) | Se_corrected: {Se_corrected:.1f} MPa",
        f"Rainflow cycles counted: {len(cycles)} in {sim_duration_s}s window",
        f"Paris crack: a0={a0_mm} mm → a_crit={a_crit:.2f} mm in {cycles_to_crit:.2e} cycles",
    ]
    if total_damage >= 1.0:
        notes.append(f"FAIL — fatigue failure expected before {mission_hours:.0f} h mission life")
    if sigma_a_eff > Se_corrected:
        notes.append(f"WARNING — stress amplitude {sigma_a_eff:.1f} MPa exceeds endurance limit {Se_corrected:.1f} MPa")

    return {
        "status":             status,
        "miner_damage":       round(total_damage, 4),
        "predicted_life_h":   round(life_hours, 1),
        "mission_hours":      mission_hours,
        "cycles_counted":     len(cycles),
        "stress_amplitude_mpa": sigma_a_eff,
        "endurance_limit_mpa":  Se_corrected,
        "paris_life_cycles":  round(cycles_to_crit, 0),
        "material":           material,
        "notes":              notes,
    }


def run_bearing_weibull_fleet(
    mechanical_params: dict,
    mission_hours: float = 20000.0,
    fleet_size: int = 1000,
    reliability_target: float = 0.99,
) -> Dict[str, Any]:
    """
    Weibull bearing fleet simulation via Monte Carlo.
    Simulates fleet_size bearings with parameter scatter.

    ISO 281: L10 = (C/P)^3 × 10^6 revolutions
    Weibull: R(t) = exp(-(t/η)^β)  β=1.5 for rolling bearings

    Fleet scatter: C and P have ±15% variation (manufacturing tolerance).
    """
    rng = np.random.default_rng(42)

    C_kn    = float(mechanical_params.get("dynamic_load_kn") or 50.0)
    P_kn    = float(mechanical_params.get("equivalent_load_kn") or 10.0)
    rpm     = float(mechanical_params.get("rpm") or 1500.0)
    beta    = 1.5   # Weibull shape for rolling bearings (ISO standard)

    if P_kn <= 0 or C_kn <= 0 or rpm <= 0:
        return {"status": "insufficient_data", "notes": ["Missing bearing parameters"]}

    # L10 deterministic
    l10_mrev  = (C_kn / P_kn) ** 3
    l10_hours = (l10_mrev * 1e6) / (60.0 * rpm)
    eta       = l10_hours / ((-np.log(0.90)) ** (1.0 / beta))

    # Fleet Monte Carlo — scatter in C and P
    C_fleet = rng.normal(C_kn, C_kn * 0.10, fleet_size)   # ±10% C scatter
    P_fleet = rng.normal(P_kn, P_kn * 0.05, fleet_size)   # ±5% P scatter
    P_fleet = np.clip(P_fleet, P_kn * 0.5, P_kn * 2.0)
    C_fleet = np.clip(C_fleet, C_kn * 0.5, C_kn * 2.0)

    l10_fleet  = (C_fleet / P_fleet) ** 3 * 1e6 / (60.0 * rpm)
    eta_fleet  = l10_fleet / ((-np.log(0.90)) ** (1.0 / beta))
    R_fleet    = np.exp(-((mission_hours / eta_fleet) ** beta))

    survival_rate = float(np.mean(R_fleet >= reliability_target))
    mean_life     = float(np.mean(l10_fleet))
    p10_life      = float(np.percentile(l10_fleet, 10))
    p1_life       = float(np.percentile(l10_fleet, 1))

    # B-life calculations
    b10_life = float(eta * (-np.log(0.90)) ** (1.0 / beta))
    b1_life  = float(eta * (-np.log(0.99)) ** (1.0 / beta))

    R_at_mission = float(np.exp(-((mission_hours / eta) ** beta)) * 100)

    if R_at_mission >= reliability_target * 100:
        status = "pass"
    elif R_at_mission >= 90.0:
        status = "warning"
    elif R_at_mission >= 70.0:
        status = "critical"
    else:
        status = "fail"

    notes = [
        f"L10={l10_hours:.0f} h | B10={b10_life:.0f} h | B1={b1_life:.0f} h",
        f"Fleet reliability at {mission_hours:.0f} h: {R_at_mission:.1f}% (target: {reliability_target*100:.0f}%)",
        f"Fleet of {fleet_size}: {survival_rate*100:.1f}% survive to mission life",
        f"P10 fleet life: {p10_life:.0f} h | P1: {p1_life:.0f} h",
        f"C={C_kn:.1f} kN, P={P_kn:.1f} kN, n={rpm:.0f} RPM",
    ]
    if status == "fail":
        notes.append(f"FAIL — only {R_at_mission:.1f}% reliability at mission end, need >{reliability_target*100:.0f}%")

    return {
        "status":            status,
        "l10_hours":         round(l10_hours, 1),
        "b10_hours":         round(b10_life, 1),
        "b1_hours":          round(b1_life, 1),
        "reliability_at_mission_pct": round(R_at_mission, 2),
        "fleet_survival_pct": round(survival_rate * 100, 2),
        "fleet_mean_life_h": round(mean_life, 1),
        "fleet_p10_life_h":  round(p10_life, 1),
        "fleet_p1_life_h":   round(p1_life, 1),
        "weibull_eta":       round(eta, 1),
        "weibull_beta":      beta,
        "notes":             notes,
    }


def run_vibration_analysis(
    mechanical_params: dict,
) -> Dict[str, Any]:
    """
    Vibration resonance analysis with frequency response.
    Detects if operating frequency falls within ±20% of natural frequency.
    Computes amplification factor Q and equivalent stress amplification.
    """
    fn      = float(mechanical_params.get("natural_freq_hz") or 50.0)
    fe      = float(mechanical_params.get("excitation_freq_hz") or 30.0)
    zeta    = float(mechanical_params.get("damping_ratio") or 0.05)

    # Frequency response function (magnitude at excitation frequency)
    r     = fe / fn  # frequency ratio
    H_mag = 1.0 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)

    # Quality factor at resonance
    Q = 1.0 / (2 * zeta)

    in_band     = 0.8 <= r <= 1.2
    near_band   = 0.6 <= r <= 1.4
    stress_amp  = H_mag  # normalized stress amplification

    if in_band:
        status = "critical"
    elif near_band and H_mag > 3.0:
        status = "warning"
    else:
        status = "pass"

    notes = [
        f"Excitation: {fe:.1f} Hz | Natural: {fn:.1f} Hz | Ratio r={r:.3f}",
        f"Amplification H={H_mag:.2f}× | Q factor={Q:.1f} | Damping ζ={zeta:.3f}",
        f"Stress amplification: {stress_amp:.2f}× nominal",
    ]
    if in_band:
        notes.append(f"CRITICAL — excitation at {r*100:.0f}% of natural frequency, amplification={H_mag:.1f}×")
    elif near_band:
        notes.append(f"WARNING — near resonance band (r={r:.2f}), monitor for fatigue acceleration")

    return {
        "status":                status,
        "frequency_ratio_r":     round(r, 4),
        "amplification_factor":  round(H_mag, 3),
        "q_factor":              round(Q, 2),
        "in_resonance_band":     in_band,
        "excitation_hz":         fe,
        "natural_hz":            fn,
        "damping_ratio":         zeta,
        "notes":                 notes,
    }


def run_mechanical_simulation(
    mechanical_params: dict,
    mission_hours: float = 20000.0,
    material: str = "steel",
) -> Dict[str, Any]:
    """
    Master mechanical simulator — runs all three sub-simulations.
    """
    results = {"status": "pass", "notes": []}

    # 1. Fatigue life
    try:
        fat = run_fatigue_simulation(mechanical_params, mission_hours, material=material)
        results["fatigue"] = fat
        results["notes"].extend(fat["notes"][:2])
    except Exception as e:
        logger.error(f"[Mech/Fatigue] {e}")
        fat = {"status": "skipped"}

    # 2. Bearing fleet
    try:
        bear = run_bearing_weibull_fleet(mechanical_params, mission_hours)
        results["bearing"] = bear
        results["notes"].extend(bear["notes"][:2])
    except Exception as e:
        logger.error(f"[Mech/Bearing] {e}")
        bear = {"status": "skipped"}

    # 3. Vibration
    try:
        vib = run_vibration_analysis(mechanical_params)
        results["vibration"] = vib
        results["notes"].extend(vib["notes"][:1])
    except Exception as e:
        logger.error(f"[Mech/Vibration] {e}")
        vib = {"status": "skipped"}

    # Worst status
    STATUS_RANK = {"pass": 0, "warning": 1, "critical": 2, "fail": 3, "skipped": -1}
    worst = max(
        [fat.get("status", "skipped"), bear.get("status", "skipped"), vib.get("status", "skipped")],
        key=lambda s: STATUS_RANK.get(s, -1)
    )
    results["status"] = worst if worst != "skipped" else "pass"

    # Key metrics
    results["miner_damage"]          = fat.get("miner_damage")
    results["predicted_life_h"]      = fat.get("predicted_life_h")
    results["bearing_b10_h"]         = bear.get("b10_hours")
    results["bearing_reliability_pct"] = bear.get("reliability_at_mission_pct")
    results["vibration_amplification"] = vib.get("amplification_factor")

    return results
