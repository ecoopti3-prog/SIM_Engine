"""
sim_router.py — Maps Idea objects to the right simulators.
Takes an Idea (dict/object), extracts params, routes to domain simulators,
assembles SimReport with cross-domain coupling analysis.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import sys, os
# Package root = sim_engine_v6/ (parent of orchestrator/)
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from core.schemas import SimReport, DomainSimResult, SimStatus
from simulators.thermal.rc_network import simulate_thermal
from simulators.electrical.power_model import simulate_electrical
from simulators.pdn.rlc_network import simulate_pdn
from simulators.data_movement.roofline import simulate_data_movement


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

def _safe_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def route_idea(idea: Dict[str, Any]) -> SimReport:
    """
    Route a single Idea (as dict) through all applicable simulators.
    Returns a complete SimReport.
    """
    t0 = time.time()
    idea_id    = idea.get("id", "unknown")
    idea_title = idea.get("title", "Untitled Idea")
    domain     = idea.get("domain", "")

    tp  = idea.get("thermal_params") or {}
    pp  = idea.get("power_params") or {}
    pdn = idea.get("pdn_params") or {}
    dm  = idea.get("data_movement_params") or {}

    domain_results: List[DomainSimResult] = []

    # ── Thermal simulation ─────────────────────────────────────────────────────────────────────────────────────
    has_thermal = any([tp, pp.get("watt"), pp.get("tdp_watt"), domain == "thermal"])
    if has_thermal:
        thermal_result = simulate_thermal(
            idea_id=idea_id,
            power_w=_safe_float(pp.get("watt") or pp.get("tdp_watt")),
            t_junction_c=_safe_float(tp.get("t_junction_c")),
            heat_flux_w_cm2=_safe_float(tp.get("heat_flux_w_cm2")),
            thermal_resistance_c_per_w=_safe_float(tp.get("thermal_resistance_c_per_w")),
            delta_t_c=_safe_float(tp.get("delta_t_c")),
            t_ambient_c=_safe_float(tp.get("t_ambient_c")),
            cop_claimed=_safe_float(tp.get("cop_claimed")),
            material=tp.get("material"),
        )
        m = thermal_result.metrics

        # ── Coupled solver: מחשב r_theta_critical האמיתי ───────────────────────
        # הסימולטור הפשוט יודע רק "חם מדי". הCoupled Solver יודע בדיוק
        # איפה המערכת קורסת ומה הגבול הקריטי — זה המספר שHypothesisGenerator צריך.
        coupled_data = {}
        try:
            from simulations.coupled.thermal_electrical_solver import run_coupled_solver
            coupled_data = run_coupled_solver(
                power_params=dict(pp),
                thermal_params=dict(tp),
                t_hotspot_c=_safe_float(m.get("T_junction_c")),
            )
            # עדכן ציון וסטאטוס על בסיס הcoupled solver — הוא מדויק יותר
            if coupled_data.get("tt_corner", {}).get("runaway"):
                thermal_result.score = 0.0
                thermal_result.status = "fail"
                r_crit = coupled_data.get("r_theta_critical", "?")
                r_act  = m.get("R_theta_c_per_w", "?")
                thermal_result.critical_failures.append(
                    f"THERMAL RUNAWAY: R_theta_actual={r_act} C/W > R_theta_critical={r_crit} C/W"
                )
            elif coupled_data.get("status") == "warning":
                if thermal_result.score > 5.0:
                    thermal_result.score = 5.0
                thermal_result.status = "marginal"
        except Exception:
            pass  # coupled solver אופציונלי — הסימולטור הפשוט ממשיך בלעדיו

        # Populate details.coupled — מה שsim_feedback_loop קורא
        # עדיפות: נתוני coupled solver (מדויקים) > נתוני הסימולטור הפשוט (אומדן)
        r_theta_actual   = m.get("R_theta_c_per_w")
        r_theta_critical = coupled_data.get("r_theta_critical")
        margin_pct       = coupled_data.get("margin_to_runaway_pct") or m.get("margin_to_jedec_pct")
        t_op             = (coupled_data.get("tt_corner") or {}).get("T_op_c")                            or m.get("T_junction_c") or m.get("T_steady_state_c")

        thermal_result.details = {
            "coupled": {
                "r_theta_actual":        r_theta_actual,
                "r_theta_critical":      r_theta_critical,
                "margin_to_runaway_pct": margin_pct,
                "tt_corner":             {"T_op_c": t_op},
                "ff_corner":             coupled_data.get("ff_corner"),
                "runaway":               coupled_data.get("tt_corner", {}).get("runaway", False),
                "solver":                "coupled_brentq" if coupled_data else "rc_analytical",
            }
        }
        if coupled_data:
            thermal_result.metrics["r_theta_critical"]      = r_theta_critical
            thermal_result.metrics["margin_to_runaway_pct"] = margin_pct
            thermal_result.metrics["T_op_coupled_c"]        = t_op

        domain_results.append(thermal_result)

    # ── Electrical simulation ──────────────────────────────────────────────────
    has_elec = any([pp, domain in ["power", "electrical"]])
    if has_elec:
        elec_result = simulate_electrical(
            idea_id=idea_id,
            power_w=_safe_float(pp.get("watt")),
            tdp_watt=_safe_float(pp.get("tdp_watt")),
            power_density_w_cm2=_safe_float(pp.get("power_density_w_cm2")),
            energy_per_op_pj=_safe_float(pp.get("energy_per_op_pj")),
            voltage_v=_safe_float(pp.get("voltage_v")),
            current_a=_safe_float(pp.get("current_a")),
            efficiency_pct=_safe_float(pp.get("efficiency_pct")),
            frequency_ghz=_safe_float(pdn.get("frequency_ghz")),
        )
        domain_results.append(elec_result)

    # ── PDN simulation ─────────────────────────────────────────────────────────
    has_pdn = any([pdn, domain == "pdn"])
    if has_pdn:
        pdn_result = simulate_pdn(
            idea_id=idea_id,
            ir_drop_mv=_safe_float(pdn.get("ir_drop_mv")),
            vdd_v=_safe_float(pdn.get("vdd_v") or pp.get("voltage_v")),
            pdn_impedance_mohm=_safe_float(pdn.get("pdn_impedance_mohm")),
            frequency_ghz=_safe_float(pdn.get("frequency_ghz")),
            bump_density_per_mm2=_safe_float(pdn.get("bump_density_per_mm2")),
            current_a=_safe_float(pdn.get("current_a") or pp.get("current_a")),
            di_dt_a_per_ns=_safe_float(pdn.get("di_dt_a_per_ns")),
            decap_nf=_safe_float(pdn.get("decap_nf")),
            n_chiplets=_safe_int(pdn.get("n_chiplets")),
        )
        domain_results.append(pdn_result)

    # ── Data movement simulation ───────────────────────────────────────────────
    has_dm = any([dm, domain == "data_movement"])
    if has_dm:
        dm_result = simulate_data_movement(
            idea_id=idea_id,
            bandwidth_gb_s=_safe_float(dm.get("bandwidth_gb_s")),
            latency_ns=_safe_float(dm.get("latency_ns")),
            memory_capacity_gb=_safe_float(dm.get("memory_capacity_gb")),
            interconnect_speed_gb_s=_safe_float(dm.get("interconnect_speed_gb_s")),
            compute_tflops=_safe_float(dm.get("compute_tflops")),
        )
        domain_results.append(dm_result)

    # ── If nothing matched domain, simulate what we can from power params ──────
    if not domain_results:
        if pp:
            domain_results.append(simulate_electrical(
                idea_id=idea_id,
                power_w=_safe_float(pp.get("watt")),
                voltage_v=_safe_float(pp.get("voltage_v")),
                energy_per_op_pj=_safe_float(pp.get("energy_per_op_pj")),
            ))
        else:
            # Nothing to simulate
            domain_results.append(DomainSimResult(
                idea_id=idea_id, domain="thermal", fidelity="L1_analytical",
                status="insufficient_data", score=4.0,
                warnings=["Idea has no numerical parameters — cannot simulate"],
                solver_notes="Idea needs ThermalParams, PowerParams, PDNParams, or DataMovementParams"
            ))

    # ── Cross-domain coupling analysis ─────────────────────────────────────────
    couplings = _analyze_cross_domain(domain_results, idea)

    # ── Aggregate score & status ───────────────────────────────────────────────
    scores   = [r.score for r in domain_results if r.status != "insufficient_data"]
    statuses = [r.status for r in domain_results]

    if not scores:
        overall_score  = 4.0
        overall_status: SimStatus = "insufficient_data"
    else:
        overall_score = round(sum(scores) / len(scores), 2)
        if any(s == "fail" for s in statuses):
            overall_status = "fail"
        elif any(s == "marginal" for s in statuses):
            overall_status = "marginal"
        elif all(s == "insufficient_data" for s in statuses):
            overall_status = "insufficient_data"
        else:
            overall_status = "pass"

    # ── Recommendation ─────────────────────────────────────────────────────────
    revision_targets: List[str] = []
    for r in domain_results:
        if r.status in ("fail", "marginal"):
            if r.falsification:
                revision_targets.append(
                    f"[{r.domain.upper()}] {r.falsification.fix_vector}"
                )
            for w in r.warnings[:2]:
                revision_targets.append(f"[{r.domain.upper()}] {w}")

    if overall_status == "fail":
        rec = "kill_physics"
    elif overall_status == "marginal":
        rec = "revise_and_resim"
    elif overall_status == "pass" and overall_score >= 7.0:
        rec = "proceed_to_prototype"
    elif overall_status == "insufficient_data":
        rec = "needs_more_data"
    else:
        rec = "revise_and_resim"

    elapsed_ms = int((time.time() - t0) * 1000)

    return SimReport(
        idea_id=idea_id,
        idea_title=idea_title,
        timestamp=datetime.now(timezone.utc).isoformat(),
        domain_results=domain_results,
        overall_status=overall_status,
        overall_score=overall_score,
        cross_domain_couplings=couplings,
        recommendation=rec,
        revision_targets=revision_targets,
        fidelity_used={r.domain: r.fidelity for r in domain_results},
        total_sim_time_ms=elapsed_ms,
    )


def _analyze_cross_domain(results: List[DomainSimResult], idea: Dict) -> List[str]:
    """Detect cross-domain couplings that neither domain simulator sees alone."""
    couplings = []
    by_domain = {r.domain: r for r in results}

    thermal = by_domain.get("thermal")
    elec    = by_domain.get("electrical")
    pdn     = by_domain.get("pdn")
    dm      = by_domain.get("data_movement")

    # Thermal ↔ Electrical coupling
    if thermal and elec:
        tj   = thermal.metrics.get("T_junction_c") or thermal.metrics.get("T_junction_computed_c")
        pd   = elec.metrics.get("power_density_w_cm2")
        if tj and pd:
            if tj > 100 and pd > 60:
                couplings.append(
                    f"THERMAL×ELECTRICAL: T_junction {tj:.0f}°C + power density {pd:.0f} W/cm² "
                    "→ electromigration lifetime < 5yr at these temperatures"
                )

    # PDN ↔ Thermal coupling
    if pdn and thermal:
        ir_pct = pdn.metrics.get("ir_drop_pct_vdd")
        tj     = thermal.metrics.get("T_junction_c") or thermal.metrics.get("T_junction_computed_c")
        if ir_pct and ir_pct > 3 and tj and tj > 100:
            couplings.append(
                f"PDN×THERMAL: IR drop {ir_pct:.1f}% at T_j {tj:.0f}°C "
                "→ leakage increases ~2× per 10°C → worsens IR drop (positive feedback loop)"
            )

    # Data Movement ↔ Thermal coupling
    if dm and thermal:
        bw = dm.metrics.get("bandwidth_gb_s")
        if bw and bw > 800:
            couplings.append(
                f"DATA×THERMAL: {bw:.0f} GB/s bandwidth → HBM3 stack thermal budget "
                "~5W per stack — must include in die-level thermal model"
            )

    # PDN ↔ Data Movement coupling
    if pdn and dm:
        n_chiplets = int(pdn.metrics.get("n_chiplets") or 1)
        bw = dm.metrics.get("bandwidth_gb_s")
        if n_chiplets > 4 and bw and bw > 1000:
            couplings.append(
                f"PDN×DATA: {n_chiplets} chiplets at {bw:.0f} GB/s → "
                "inter-chiplet ground bounce couples into signal integrity"
            )

    return couplings
