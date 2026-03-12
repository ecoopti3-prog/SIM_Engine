"""
main.py — Entry point for the Physics Simulation Engine.

Usage:
  python main.py --diamonds              # simulate all Diamond ideas
  python main.py --idea <idea_id>        # simulate one specific idea
  python main.py --demo                  # run with synthetic idea (no Supabase needed)
  python main.py --schema                # print Supabase SQL schema for sim_results
"""
from __future__ import annotations
import argparse, logging, sys, time, json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sim_engine")


DEMO_IDEA = {
    "id":     "demo-0000-0000-0000-000000000001",
    "title":  "Graphene TIM for 3nm AI Accelerator — 2x heat flux improvement",
    "domain": "thermal",
    "problem_statement": "Current TIM1 limits heat flux to ~80 W/cm². Proposed graphene TIM achieves ~160 W/cm².",
    "physical_limit":    "Copper practical limit ~100 W/cm². Graphene theoretical ~500 W/cm².",
    "diamond_score":     8.7,
    "power_params": {
        "watt":              400.0,
        "tdp_watt":          450.0,
        "power_density_w_cm2": 90.0,
        "voltage_v":          0.75,
        "current_a":         533.0,
        "energy_per_op_pj":   0.01,
    },
    "thermal_params": {
        "t_junction_c":                 105.0,
        "heat_flux_w_cm2":               85.0,
        "thermal_resistance_c_per_w":    0.20,
        "t_ambient_c":                   40.0,
        "material":                    "graphene",
    },
    "pdn_params": {
        "vdd_v":               0.75,
        "current_a":           533.0,
        "ir_drop_mv":           18.0,
        "pdn_impedance_mohm":    3.5,
        "bump_density_per_mm2": 2500.0,
        "di_dt_a_per_ns":        8.0,
        "decap_nf":             800.0,
        "frequency_ghz":          2.0,
    },
    "data_movement_params": {
        "bandwidth_gb_s":   3600.0,
        "compute_tflops":   2000.0,
        "latency_ns":         30.0,
    },
}


def run_demo() -> None:
    logger.info("Running DEMO simulation on synthetic Diamond idea...")
    from core.dispatcher import dispatch
    from reports.generator import generate_report

    result = dispatch(DEMO_IDEA)

    report_path = generate_report(result, output_base="reports/output")
    result.report_path = report_path

    print("\n" + "="*60)
    print(f"  SIMULATION COMPLETE")
    print(f"  Idea: {result.idea_title}")
    print(f"  Overall: {result.overall_status.upper()} | Score: {result.sim_score}/10")
    print("="*60)
    if result.critical_failures:
        print("\n  ⛔ CRITICAL FAILURES:")
        for f in result.critical_failures:
            print(f"     • {f}")
    print("\n  KEY INSIGHTS:")
    for insight in result.key_insights[:6]:
        print(f"     • {insight}")
    print(f"\n  📄 Report: {report_path}")
    print("="*60 + "\n")

    # Print per-domain summary
    if result.thermal:
        r = result.thermal
        hotspot_str = f" | hotspot={r.t_hotspot_ss_c}°C (+{r.hotspot_excess_c}°C)" if r.t_hotspot_ss_c else ""
        print(f"  🌡  Thermal   [{r.status.upper():8}] T_j={r.t_junction_ss_c}°C | margin={r.thermal_margin_pct}%{hotspot_str}")
    if result.coupled:
        r = result.coupled
        if r.tt_runaway:
            print(f"  🔗  Coupled   [{r.status.upper():8}] RUNAWAY (TT) | R_crit={r.r_theta_critical} C/W | actual={r.r_theta_actual} C/W")
        else:
            ff_str = "FF:RUNAWAY" if r.ff_runaway else f"FF:{r.ff_T_op_c}°C"
            print(f"  🔗  Coupled   [{r.status.upper():8}] T_op={r.tt_T_op_c}°C | factor={r.tt_runaway_factor:.3f} | {ff_str} | margin={r.margin_to_runaway_pct:.0f}%")
    if result.pdn:
        r = result.pdn
        ssn_str = f" | SSN={r.ssn_at_50pct_mv:.0f}mV@50%" if r.ssn_at_50pct_mv else ""
        print(f"  ⚡  PDN       [{r.status.upper():8}] droop={r.droop_mv}mV ({r.droop_pct:.1f}%){ssn_str}")
    if result.electrical:
        r = result.electrical
        pvt_str = f" | PVT:{r.pvt_status} slack={r.pvt_min_slack_ps:.0f}ps" if r.pvt_min_slack_ps is not None else ""
        print(f"  🔋  Electrical[{r.status.upper():8}] T_op={r.t_equilibrium_c}°C | P_leak={r.p_leakage_w:.0f}W{pvt_str}")
    if result.aging:
        r = result.aging
        print(f"  ⏳  Aging     [{r.status.upper():8}] NBTI={r.nbti_delta_vt_mv:.1f}mV (+{r.nbti_delta_t_pd_ps:.0f}ps) | EM={r.em_mttf_years:.0f}yr | duty={r.duty_cycle:.0%}")
    if result.monte_carlo:
        r = result.monte_carlo
        top = f" | top_param={r.top_sensitivity_param}" if r.top_sensitivity_param else ""
        jedec_str = f" | JEDEC_10yr={r.fleet_pct_jedec:.0f}%" if r.fleet_pct_jedec is not None else ""
        print(f"  🎲  MC Yield  [{r.status.upper():8}] yield={r.yield_pct:.1f}% ±{r.yield_ci_pct/2:.1f}% | N={r.n_samples}{top}{jedec_str}")
    if result.data_movement:
        r = result.data_movement
        noc_str = f" | NoC:{r.noc_efficiency_pct:.0f}%eff" if r.noc_efficiency_pct else ""
        print(f"  📡  Data Mov  [{r.status.upper():8}] roofline={r.efficiency_pct:.0f}%{noc_str} | bw={r.noc_bw_effective_gb_s or r.bandwidth_utilization}GB/s")
    print()


def run_diamonds(limit: int = 20) -> None:
    from db.reader import load_diamond_ideas
    from core.dispatcher import dispatch
    from reports.generator import generate_report
    import os
    from dotenv import load_dotenv
    load_dotenv()
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    ideas = load_diamond_ideas(limit=limit)
    if not ideas:
        logger.warning("No Diamond ideas found in Supabase.")
        return

    logger.info(f"Simulating {len(ideas)} Diamond ideas...")
    from output.rd_bridge import push_sim_results_to_supabase
    from core.schemas import SimReport, DomainSimResult
    import time
    from datetime import datetime, timezone
    cycle_id = f"sim-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    reports = []
    for i, idea in enumerate(ideas):
        logger.info(f"[{i+1}/{len(ideas)}] {idea['id'][:8]} — {idea.get('title','')[:60]}")
        result = dispatch(idea)
        report_path = generate_report(result)
        logger.info(f"  → {result.overall_status.upper()} score={result.sim_score} report={report_path}")
        # Build a minimal SimReport for rd_bridge
        dr = []
        if result.thermal:
            c = result.coupled
            coupled_details = {}
            if c:
                coupled_details = {"r_theta_actual": c.r_theta_actual, "r_theta_critical": c.r_theta_critical,
                    "margin_to_runaway_pct": c.margin_to_runaway_pct, "tt_corner": {"T_op_c": c.tt_T_op_c},
                    "ff_corner": {"T_op_c": c.ff_T_op_c, "runaway": c.ff_runaway}, "runaway": c.tt_runaway}
            dr.append(DomainSimResult(idea_id=idea["id"], domain="thermal",
                status=result.thermal.status, score=result.sim_score,
                details={"coupled": coupled_details}))
        rec = "kill_physics" if result.sim_score < 3.0 else ("proceed_to_prototype" if result.sim_score >= 7.0 else "revise_and_resim")
        reports.append(SimReport(idea_id=idea["id"], idea_title=idea.get("title",""),
            domain_results=dr, overall_status=result.overall_status,
            overall_score=result.sim_score, recommendation=rec,
            revision_targets=result.critical_failures[:3], duration_ms=result.duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat()))
    if supabase_url and supabase_key:
        push_sim_results_to_supabase(reports, supabase_url, supabase_key, cycle_id)
    else:
        logger.warning("SUPABASE_URL/KEY not set — results not pushed to DB. Use run_sim_on_diamonds.py for production.")


def run_single(idea_id: str) -> None:
    from db.reader import load_diamond_ideas
    from core.dispatcher import dispatch
    from reports.generator import generate_report

    ideas = load_diamond_ideas(limit=100)
    idea  = next((i for i in ideas if i["id"] == idea_id or i["id"].startswith(idea_id)), None)
    if not idea:
        logger.error(f"Idea {idea_id} not found in Diamond ideas")
        return

    result = dispatch(idea)
    report_path = generate_report(result)
    logger.info(f"Done: {result.overall_status} score={result.sim_score} report={report_path}")
    logger.info("Note: Use run_sim_on_diamonds.py to push results to Supabase.")


def main():
    parser = argparse.ArgumentParser(description="Physics Simulation Engine")
    parser.add_argument("--demo",     action="store_true", help="Run demo with synthetic idea")
    parser.add_argument("--diamonds", action="store_true", help="Simulate all Diamond ideas from Supabase")
    parser.add_argument("--idea",     type=str,            help="Simulate specific idea by ID")
    parser.add_argument("--schema",   action="store_true", help="Print Supabase SQL for sim_results table")
    parser.add_argument("--limit",    type=int, default=20, help="Max ideas to simulate")
    args = parser.parse_args()

    if args.schema:
        from db.reader import get_sim_schema_sql
        print(get_sim_schema_sql())
        return

    if args.demo:
        run_demo()
        return

    if args.diamonds:
        run_diamonds(limit=args.limit)
        return

    if args.idea:
        run_single(args.idea)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
