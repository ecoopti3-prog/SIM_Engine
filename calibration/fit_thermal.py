"""
fit_thermal.py — Physical Calibration Fitter

Reads serial data from Arduino stress test, fits the thermal RC model,
and extracts calibrated R_theta and C_thermal constants.

USAGE:
  # Live mode (Arduino connected):
  python calibration/fit_thermal.py --port /dev/ttyACM0

  # From saved data file:
  python calibration/fit_thermal.py --file calibration/data/run_001.csv

  # Demo mode (synthetic data with known answer):
  python calibration/fit_thermal.py --demo

OUTPUT:
  calibration/results/calibration_result.json   — calibrated constants
  calibration/results/thermal_fit.png            — measured vs simulated plot
  
  Updates:  config/calibration_constants.py      — injected into sim_engine

MODEL FITTED:
  T(t) = T_ss - ΔT × exp(-t/τ)
  where:
    T_ss = steady-state temperature (fitted)
    ΔT   = T_ss - T_0 (temperature rise)
    τ    = R_theta × C_thermal (time constant, fitted)
  
  From fitted params:
    R_theta    = ΔT / ΔP   [°C/W]
    C_thermal  = τ / R_theta [J/K]
    
  Comparison with datasheet:
    ATmega328P DIP28: R_theta_ja = 68°C/W (published JEDEC value)
    
  The RATIO (fitted/datasheet) becomes the calibration correction factor
  applied to all subsequent simulations.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import json, os, sys, argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# ── ATmega328P reference data (JEDEC + datasheet) ────────────────────────────
ATMEGA_REF = {
    "r_theta_ja_published":  68.0,   # °C/W (DIP28 JEDEC)
    "p_active_mw":           75.0,   # mW at 5V, 16MHz (datasheet)
    "p_idle_mw":             27.5,   # mW in idle mode
    "delta_p_mw":            47.5,   # mW = 75 - 27.5
    "delta_t_expected_c":    3.23,   # °C = 0.0475W × 68°C/W
    "tau_expected_s":        54.4,   # s = R × C = 68 × 0.8
    "c_thermal_ref":          0.8,   # J/K (estimated)
}


def _thermal_rise(t: np.ndarray, T_0: float, T_ss: float, tau: float) -> np.ndarray:
    """T(t) = T_ss - (T_ss - T_0) × exp(-t/τ)"""
    return T_ss - (T_ss - T_0) * np.exp(-t / np.maximum(tau, 0.1))


def _parse_serial_log(lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse CAL,timestamp_ms,phase,temp_x100,mode_int,loop_count,elapsed_s lines.
    Returns (t_idle_s, T_idle_c, t_active_s, T_active_c).
    """
    t_idle, T_idle, t_active, T_active = [], [], [], []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("CAL,"):
            continue
        try:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            phase   = parts[2].strip()
            temp_c  = int(parts[3]) / 100.0
            elapsed = float(parts[6]) if len(parts) > 6 else float(parts[1]) / 1000.0
            
            if phase == "IDLE":
                t_idle.append(elapsed)
                T_idle.append(temp_c)
            elif phase == "ACTIVE":
                t_active.append(elapsed)
                T_active.append(temp_c)
        except (ValueError, IndexError):
            continue
    
    return (np.array(t_idle), np.array(T_idle),
            np.array(t_active), np.array(T_active))


def _smooth(T: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing for noisy sensor data."""
    if len(T) < window:
        return T
    window = min(window, len(T) if len(T) % 2 == 1 else len(T) - 1)
    window = window if window % 2 == 1 else window - 1
    return savgol_filter(T, max(3, window), 2)


def fit_rc_model(
    t_active: np.ndarray,
    T_active: np.ndarray,
    T_idle_mean: float,
    delta_p_mw: float = ATMEGA_REF["delta_p_mw"],
) -> Dict:
    """
    Fit T(t) = T_ss - ΔT × exp(-t/τ) to active-phase data.
    
    Returns fitted R_theta, C_thermal, and quality metrics.
    """
    T_active_smooth = _smooth(T_active)
    
    # Initial parameter estimates
    T_0_init  = T_active_smooth[0] if len(T_active_smooth) > 0 else T_idle_mean
    T_ss_init = T_active_smooth[-1] if len(T_active_smooth) > 0 else T_idle_mean + 3.0
    tau_init  = 50.0   # seconds (close to expected 54.4s)
    
    try:
        popt, pcov = curve_fit(
            _thermal_rise,
            t_active,
            T_active_smooth,
            p0=[T_0_init, T_ss_init, tau_init],
            bounds=(
                [T_idle_mean - 5,  T_idle_mean,     1.0],   # lower
                [T_idle_mean + 5,  T_idle_mean + 20, 300.0]  # upper
            ),
            maxfev=10000,
        )
        
        T_0_fit, T_ss_fit, tau_fit = popt
        perr = np.sqrt(np.diag(pcov))
        
        delta_t_measured = T_ss_fit - T_idle_mean
        delta_p_w        = delta_p_mw * 1e-3
        r_theta_measured = delta_t_measured / delta_p_w
        c_thermal_measured = tau_fit / r_theta_measured
        
        # R² goodness of fit
        T_pred = _thermal_rise(t_active, *popt)
        ss_res = np.sum((T_active_smooth - T_pred) ** 2)
        ss_tot = np.sum((T_active_smooth - T_active_smooth.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return {
            "success":           True,
            "T_0_fitted_c":      round(float(T_0_fit), 3),
            "T_ss_fitted_c":     round(float(T_ss_fit), 3),
            "tau_fitted_s":      round(float(tau_fit), 2),
            "T_idle_mean_c":     round(float(T_idle_mean), 3),
            "delta_T_c":         round(float(delta_t_measured), 4),
            "r_theta_measured":  round(float(r_theta_measured), 2),
            "c_thermal_measured": round(float(c_thermal_measured), 4),
            "r_squared":         round(float(r_squared), 5),
            "param_stderr":      [round(float(e), 4) for e in perr],
            "n_points":          len(t_active),
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def compute_correction_factors(fit: Dict) -> Dict:
    """
    Compute how far the real silicon is from datasheet spec.
    These factors calibrate all future simulations.
    
    correction_factor = measured / published
    >1.0 → chip runs hotter than spec (bad thermal contact, higher-than-spec P)
    <1.0 → chip runs cooler than spec (better package, lower actual P)
    """
    if not fit["success"]:
        return {}
    
    ref = ATMEGA_REF
    r_factor = fit["r_theta_measured"] / ref["r_theta_ja_published"]
    c_factor = fit["c_thermal_measured"] / ref["c_thermal_ref"]
    tau_factor = fit["tau_fitted_s"] / ref["tau_expected_s"]
    
    # Estimate actual power from ΔT / R_theta_measured
    p_actual_mw = fit["delta_T_c"] / fit["r_theta_measured"] * 1000
    p_factor    = p_actual_mw / ref["delta_p_mw"]
    
    quality = "EXCELLENT" if fit["r_squared"] > 0.99 else (
              "GOOD"      if fit["r_squared"] > 0.95 else (
              "MARGINAL"  if fit["r_squared"] > 0.85 else "POOR"))
    
    return {
        "r_theta_correction_factor": round(r_factor, 4),
        "c_thermal_correction_factor": round(c_factor, 4),
        "tau_correction_factor": round(tau_factor, 4),
        "power_correction_factor": round(p_factor, 4),
        "fit_quality": quality,
        "r_theta_published": ref["r_theta_ja_published"],
        "r_theta_measured": fit["r_theta_measured"],
        "c_thermal_published": ref["c_thermal_ref"],
        "c_thermal_measured": fit["c_thermal_measured"],
        "interpretation": _interpret(r_factor, c_factor),
    }


def _interpret(r_factor: float, c_factor: float) -> str:
    lines = []
    if abs(r_factor - 1.0) < 0.10:
        lines.append("Thermal resistance within 10% of datasheet (good calibration).")
    elif r_factor > 1.10:
        lines.append(f"R_theta {r_factor:.1f}x higher than spec — possible: poor airflow, "
                     f"sensor not close enough to chip, or ambient temp drift.")
    else:
        lines.append(f"R_theta {r_factor:.1f}x lower than spec — possible: forced airflow "
                     f"present, or sensor measuring ambient rather than junction.")
    
    if abs(c_factor - 1.0) < 0.30:
        lines.append("Thermal capacitance reasonable for DIP28 package.")
    elif c_factor > 1.5:
        lines.append("High thermal capacitance — sensor may be measuring board+chip combined.")
    return " ".join(lines)


def generate_demo_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic ground-truth data for testing without Arduino.
    Uses ATMEGA_REF constants + realistic noise (±0.1°C = LM35 precision).
    """
    rng = np.random.default_rng(42)
    ref = ATMEGA_REF
    
    # Idle: 120 seconds, stable at 25°C ± 0.1°C noise
    t_idle  = np.arange(0, 120, 5, dtype=float)
    T_idle  = 25.0 + rng.normal(0, 0.08, len(t_idle))
    
    # Active: 480 seconds of thermal rise
    t_act   = np.arange(0, 480, 5, dtype=float)
    T_act   = _thermal_rise(t_act, 25.0, 25.0 + ref["delta_t_expected_c"], ref["tau_expected_s"])
    T_act   = T_act + rng.normal(0, 0.08, len(T_act))
    
    return t_idle, T_idle, t_act, T_act


def run_calibration(
    port: Optional[str] = None,
    file: Optional[str] = None,
    demo: bool = False,
    output_dir: str = "calibration/results",
) -> Dict:
    """Main calibration entry point."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ── Load data ─────────────────────────────────────────────────────────
    if demo:
        print("[DEMO] Using synthetic ATmega328P data (known answer: R_theta=68°C/W, τ=54.4s)")
        t_idle, T_idle, t_active, T_active = generate_demo_data()
        source = "synthetic_demo"
        
    elif file:
        print(f"[FILE] Loading calibration data from {file}")
        with open(file) as f:
            lines = f.readlines()
        t_idle, T_idle, t_active, T_active = _parse_serial_log(lines)
        source = f"file:{file}"
        
    elif port:
        print(f"[SERIAL] Reading from {port} at 115200 baud...")
        print("         Waiting for calibration run to complete (~12 minutes)...")
        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            lines = []
            in_run = False
            while True:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if "START" in line:
                    in_run = True
                    print("  → Calibration started")
                if in_run:
                    lines.append(line)
                    if line.startswith("CAL,"):
                        parts = line.split(",")
                        print(f"  → {parts[2]:6} t={parts[6]:4}s T={int(parts[3])/100:.2f}°C")
                if "COOLING_DOWN" in line:
                    print("  → Run complete")
                    break
            ser.close()
        except ImportError:
            print("ERROR: pyserial not installed. Run: pip install pyserial")
            sys.exit(1)
        t_idle, T_idle, t_active, T_active = _parse_serial_log(lines)
        source = f"serial:{port}"
    else:
        print("No data source. Using --demo mode.")
        t_idle, T_idle, t_active, T_active = generate_demo_data()
        source = "demo_fallback"
    
    # ── Validate data ─────────────────────────────────────────────────────
    if len(t_idle) < 3 or len(t_active) < 10:
        print(f"ERROR: Insufficient data. Idle points: {len(t_idle)}, Active: {len(t_active)}")
        sys.exit(1)
    
    T_idle_mean  = float(np.mean(T_idle))
    T_idle_std   = float(np.std(T_idle))
    T_active_max = float(np.max(T_active))
    
    print(f"\n[DATA] Idle: n={len(t_idle)}, T_mean={T_idle_mean:.2f}°C ± {T_idle_std:.3f}°C")
    print(f"[DATA] Active: n={len(t_active)}, T_max={T_active_max:.2f}°C, ΔT={T_active_max - T_idle_mean:.2f}°C")
    
    # ── Fit RC model ──────────────────────────────────────────────────────
    print("\n[FIT] Fitting T(t) = T_ss - ΔT × exp(-t/τ)...")
    fit = fit_rc_model(t_active, T_active, T_idle_mean)
    
    if not fit["success"]:
        print(f"[FIT] FAILED: {fit.get('error')}")
        print("      Check: is sensor near the ATmega328P? Is data from full run?")
        sys.exit(1)
    
    print(f"[FIT] T_ss={fit['T_ss_fitted_c']:.2f}°C, τ={fit['tau_fitted_s']:.1f}s, R²={fit['r_squared']:.4f}")
    
    # ── Compute correction factors ────────────────────────────────────────
    factors = compute_correction_factors(fit)
    
    print(f"\n[CALIBRATION RESULT]")
    print(f"  R_theta:    measured={fit['r_theta_measured']:.1f}°C/W  |  datasheet={ATMEGA_REF['r_theta_ja_published']:.0f}°C/W  |  factor={factors['r_theta_correction_factor']:.3f}×")
    print(f"  C_thermal:  measured={fit['c_thermal_measured']:.3f}J/K  |  reference={ATMEGA_REF['c_thermal_ref']:.1f}J/K   |  factor={factors['c_thermal_correction_factor']:.3f}×")
    print(f"  τ:          measured={fit['tau_fitted_s']:.1f}s    |  expected={ATMEGA_REF['tau_expected_s']:.1f}s      |  factor={factors['tau_correction_factor']:.3f}×")
    print(f"  Fit quality: {factors['fit_quality']} (R²={fit['r_squared']:.4f})")
    print(f"  → {factors['interpretation']}")
    
    # ── Save results ──────────────────────────────────────────────────────
    result = {
        "timestamp":       datetime.now().isoformat(),
        "source":          source,
        "chip":            "ATmega328P_DIP28",
        "fit":             fit,
        "correction_factors": factors,
        "reference":       ATMEGA_REF,
        "t_idle":          t_idle.tolist(),
        "T_idle":          T_idle.tolist(),
        "t_active":        t_active.tolist(),
        "T_active":        T_active.tolist(),
    }
    
    result_path = f"{output_dir}/calibration_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] {result_path}")
    
    # ── Update sim_engine calibration constants ───────────────────────────
    _write_calibration_constants(factors, output_dir)
    
    # ── Optional plot ─────────────────────────────────────────────────────
    _plot_fit(t_idle, T_idle, t_active, T_active, fit, factors, output_dir)
    
    return result


def _write_calibration_constants(factors: Dict, output_dir: str) -> None:
    """Write calibration_constants.py for import by sim_engine modules."""
    calib_path = "config/calibration_constants.py"
    content = f'''"""
calibration_constants.py — Auto-generated by fit_thermal.py
Generated: {datetime.now().isoformat()}

These constants are correction factors measured on REAL SILICON (ATmega328P)
and applied to all thermal/electrical simulations.

Factor = measured_on_silicon / datasheet_value
  >1.0 → real chip is worse than spec (conservative)
  <1.0 → real chip is better than spec
"""

# Thermal RC model corrections
R_THETA_CORRECTION     = {factors["r_theta_correction_factor"]:.4f}   # measured/published R_theta
C_THERMAL_CORRECTION   = {factors["c_thermal_correction_factor"]:.4f}  # measured/reference C_thermal
TAU_CORRECTION         = {factors["tau_correction_factor"]:.4f}   # measured/expected tau

# Power model corrections  
POWER_CORRECTION       = {factors["power_correction_factor"]:.4f}   # actual/assumed power

# Source chip
CALIBRATED_ON          = "ATmega328P_DIP28"
CALIBRATION_R_MEASURED = {factors["r_theta_measured"]:.2f}   # °C/W measured
FIT_QUALITY            = "{factors["fit_quality"]}"

def apply_thermal(r_theta: float) -> float:
    """Apply correction to any R_theta estimate."""
    return r_theta * R_THETA_CORRECTION

def apply_power(p_w: float) -> float:
    """Apply correction to power estimate."""
    return p_w * POWER_CORRECTION
'''
    os.makedirs("config", exist_ok=True)
    with open(calib_path, "w") as f:
        f.write(content)
    print(f"[SAVED] {calib_path} (correction factors injected into sim_engine)")


def _plot_fit(t_idle, T_idle, t_active, T_active, fit, factors, output_dir):
    """Generate measured vs. fitted vs. expected (datasheet) comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        fig.suptitle("Arduino Thermal Calibration — Measured vs Simulated\n"
                     "ATmega328P DIP28 @ 5V/16MHz", fontsize=13, fontweight="bold")

        # ── Top panel: temperature vs time ─────────────────────────────
        ax = axes[0]
        t_offset = t_active[0] if len(t_active) > 0 else 0
        
        ax.scatter(t_idle, T_idle, s=12, c="#2196F3", alpha=0.6, label="Measured (IDLE)", zorder=5)
        ax.scatter(t_active, T_active, s=12, c="#F44336", alpha=0.6, label="Measured (ACTIVE)", zorder=5)
        
        if fit["success"]:
            t_fit = np.linspace(0, max(t_active) if len(t_active) > 0 else 480, 500)
            T_fit = _thermal_rise(t_fit, fit["T_0_fitted_c"], fit["T_ss_fitted_c"], fit["tau_fitted_s"])
            ax.plot(t_fit, T_fit, "r-", lw=2.5, label=f"Fitted model (τ={fit['tau_fitted_s']:.0f}s, R²={fit['r_squared']:.4f})", zorder=4)
            
            # Expected from datasheet
            ref = ATMEGA_REF
            T_expected = _thermal_rise(t_fit, fit["T_idle_mean_c"],
                                       fit["T_idle_mean_c"] + ref["delta_t_expected_c"], ref["tau_expected_s"])
            ax.plot(t_fit, T_expected, "g--", lw=1.5, alpha=0.7,
                    label=f"Datasheet prediction (R_theta={ref['r_theta_ja_published']}°C/W)", zorder=3)
            
            # Annotate T_ss
            ax.axhline(fit["T_ss_fitted_c"], color="red", lw=1, ls=":", alpha=0.5)
            ax.annotate(f"T_ss={fit['T_ss_fitted_c']:.2f}°C\n(ΔT={fit['delta_T_c']:.2f}°C)",
                        xy=(max(t_active)*0.7, fit["T_ss_fitted_c"]),
                        xytext=(max(t_active)*0.5, fit["T_ss_fitted_c"] + 0.3),
                        fontsize=9, color="darkred",
                        arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8))
        
        ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.5, label="ACTIVE start")
        ax.set_xlabel("Time in ACTIVE phase [s]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_title("Temperature vs Time (IDLE baseline + ACTIVE stress)")
        ax.legend(loc="center right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Bottom panel: calibration summary bar chart ─────────────────
        ax2 = axes[1]
        metrics = ["R_theta\n[°C/W]", "C_thermal\n[J/K]", "τ\n[s]"]
        measured = [fit["r_theta_measured"], fit["c_thermal_measured"], fit["tau_fitted_s"]]
        expected = [ATMEGA_REF["r_theta_ja_published"], ATMEGA_REF["c_thermal_ref"], ATMEGA_REF["tau_expected_s"]]
        
        x = np.arange(len(metrics))
        width = 0.35
        bars_m = ax2.bar(x - width/2, measured, width, label="Measured", color="#F44336", alpha=0.8)
        bars_e = ax2.bar(x + width/2, expected, width, label="Datasheet/Reference", color="#2196F3", alpha=0.8)
        
        # Add correction factor labels
        for i, (m, e) in enumerate(zip(measured, expected)):
            factor = m / e if e != 0 else 0
            color  = "darkred" if abs(factor - 1.0) > 0.15 else "darkgreen"
            ax2.text(x[i], max(m, e) * 1.05, f"{factor:.2f}×", 
                     ha="center", fontsize=10, fontweight="bold", color=color)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.set_title(f"Calibration Summary — Fit quality: {factors['fit_quality']}")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylabel("Value")
        
        plt.tight_layout()
        plot_path = f"{output_dir}/thermal_fit.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {plot_path}")
    except Exception as e:
        print(f"[PLOT] Skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arduino Thermal Calibration Fitter")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--port", help="Serial port (e.g., /dev/ttyACM0 or COM3)")
    group.add_argument("--file", help="Pre-recorded CSV file from Arduino")
    group.add_argument("--demo", action="store_true", help="Synthetic demo data (no hardware needed)")
    parser.add_argument("--output-dir", default="calibration/results")
    args = parser.parse_args()
    
    run_calibration(port=args.port, file=args.file, demo=args.demo, output_dir=args.output_dir)
