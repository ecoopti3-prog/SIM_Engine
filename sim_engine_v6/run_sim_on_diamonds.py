"""
run_sim_on_diamonds.py — Entry point for GitHub Actions sim pipeline.

Loads new diamond ideas from Supabase (ideas not yet simulated or near-miss iterations),
runs full Sim Engine pipeline, pushes results back via sim_feedback_loop (rd_bridge v7).

RESEARCH LOOP:
  Normal diamonds: run once after Cycle 4.
  Near-miss diamonds: re-run after each Research Loop iteration (iteration_count > 0).
  Max iterations: 3 (configurable). After 3 cycles of near-miss without improvement → archive.
"""
from __future__ import annotations
import argparse
import logging
import sys
import os
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sim_runner")

MAX_NEAR_MISS_ITERATIONS = 3


def load_unsimulated_diamonds(sb, hours_back: int = 26) -> list:
    """Load diamonds that haven't been simulated yet (or need re-simulation)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    try:
        # Ideas that are diamonds but have no sim_score yet
        res = sb.table("ideas") \
            .select("*") \
            .in_("status", ["diamond", "active"]) \
            .is_("sim_score", "null") \
            .gte("created_at", cutoff) \
            .execute()
        new_diamonds = res.data or []
        logger.info(f"[SimRunner] {len(new_diamonds)} new unsimulated diamonds/ideas found")
        return new_diamonds
    except Exception as e:
        logger.error(f"[SimRunner] load_unsimulated_diamonds failed: {e}")
        return []


def load_near_miss_for_reiteration(sb) -> list:
    """
    Load near-miss ideas that have been through Research Loop
    but haven't been re-simulated after the new focused searches.
    These are ideas where iteration_count > 0 and last_sim_at is older than 12h.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    try:
        res = sb.table("ideas") \
            .select("*") \
            .eq("near_miss", True) \
            .in_("status", ["active", "diamond"]) \
            .gt("iteration_count", 0) \
            .lt("iteration_count", MAX_NEAR_MISS_ITERATIONS) \
            .or_(f"last_sim_at.is.null,last_sim_at.lt.{cutoff}") \
            .execute()
        near_miss = res.data or []
        logger.info(f"[SimRunner] {len(near_miss)} near-miss ideas ready for re-simulation")
        return near_miss
    except Exception as e:
        logger.warning(f"[SimRunner] load_near_miss_for_reiteration failed (non-blocking): {e}")
        return []


def run(idea_id: str = "", force_all: bool = False):
    from dotenv import load_dotenv
    load_dotenv()

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        logger.error("[SimRunner] SUPABASE_URL and SUPABASE_KEY required")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(supabase_url, supabase_key)

    # Load ideas to simulate
    if idea_id:
        res = sb.table("ideas").select("*").eq("id", idea_id).execute()
        ideas = res.data or []
        logger.info(f"[SimRunner] Specific idea: {len(ideas)} found for id={idea_id[:8]}")
    elif force_all:
        res = sb.table("ideas").select("*").in_("status", ["diamond"]).execute()
        ideas = res.data or []
        logger.info(f"[SimRunner] Force all: {len(ideas)} diamonds")
    else:
        ideas = load_unsimulated_diamonds(sb) + load_near_miss_for_reiteration(sb)
        # Deduplicate
        seen = set()
        deduped = []
        for i in ideas:
            if i["id"] not in seen:
                seen.add(i["id"])
                deduped.append(i)
        ideas = deduped

    if not ideas:
        logger.info("[SimRunner] No ideas to simulate. Exiting cleanly.")
        return

    logger.info(f"[SimRunner] Simulating {len(ideas)} ideas...")

    # Import sim pipeline
    # Package root = directory containing this script (sim_engine_v6/)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    # Use dispatcher (full pipeline: coupled, PVT, aging, Monte Carlo)
    # instead of sim_router (basic 4-domain only)
    from core.dispatcher import dispatch as _dispatch_full

    def route_idea(idea):
        """
        Wrapper: runs the full dispatcher pipeline and converts SimResult → SimReport
        so that sim_feedback_loop (which expects SimReport format) works correctly.
        """
        from core.schemas import SimResult, SimReport, DomainSimResult
        result: SimResult = _dispatch_full(idea)

        # Build domain_results list with details.coupled populated for sim_feedback_loop
        domain_results = []
        if result.thermal:
            coupled_details = {}
            if result.coupled:
                c = result.coupled
                coupled_details = {
                    "r_theta_actual":        c.r_theta_actual,
                    "r_theta_critical":      c.r_theta_critical,
                    "margin_to_runaway_pct": c.margin_to_runaway_pct,
                    "tt_corner":             {"T_op_c": c.tt_T_op_c},
                    "ff_corner":             {"T_op_c": c.ff_T_op_c, "runaway": c.ff_runaway},
                    "runaway":               c.tt_runaway,
                    "solver":                "coupled_brentq",
                }
            domain_results.append({
                "domain":  "thermal",
                "status":  result.thermal.status,
                "details": {"coupled": coupled_details},
            })

        if result.pdn:
            domain_results.append({"domain": "pdn", "status": result.pdn.status, "details": {}})
        if result.electrical:
            domain_results.append({"domain": "electrical", "status": result.electrical.status, "details": {}})
        if result.data_movement:
            domain_results.append({"domain": "data_movement", "status": result.data_movement.status, "details": {}})
        if result.monte_carlo:
            domain_results.append({
                "domain":  "yield",
                "status":  result.monte_carlo.status,
                "details": {
                    "yield_pct":             result.monte_carlo.yield_pct,
                    "top_sensitivity_param": result.monte_carlo.top_sensitivity_param,
                },
            })
        if result.aging:
            mttf_domain = "em" if (result.aging.em_mttf_ok is False) else "nbti"
            domain_results.append({
                "domain":  "reliability",
                "status":  result.aging.status,
                "details": {
                    "min_mttf_years":    result.aging.min_mttf_years,
                    "top_failure_domain": mttf_domain,
                },
            })

        # Infer recommendation from sim score and status
        score = result.sim_score
        status = result.overall_status
        if status == "fail" or score < 3.0:
            rec = "kill_physics"
        elif status in ("warning", "critical") or score < 6.5:
            rec = "revise_and_resim"
        elif score >= 7.0:
            rec = "proceed_to_prototype"
        else:
            rec = "revise_and_resim"

        # Return a plain dict — sim_feedback_loop.push_sim_result_to_db expects dict
        return type("SimReportCompat", (), {
            "idea_id":        result.idea_id,
            "overall_status": status,
            "overall_score":  score,
            "recommendation": rec,
            "revision_targets": result.critical_failures[:5],
            "domain_results": domain_results,
            "duration_ms":    result.duration_ms,
            "top_sensitivity_param": (
                result.monte_carlo.top_sensitivity_param if result.monte_carlo else None
            ),
            "model_dump": lambda mode=None: {
                "overall_status": status,
                "overall_score":  score,
                "recommendation": rec,
                "revision_targets": result.critical_failures[:5],
                "domain_results": domain_results,
                "duration_ms":    result.duration_ms,
                "top_sensitivity_param": (
                    result.monte_carlo.top_sensitivity_param if result.monte_carlo else None
                ),
            },
        })()
    # sim_feedback_loop is the real push module (sim_feedback_push was a stale reference)

    results = {"pass": 0, "fail": 0, "marginal": 0, "near_miss": 0, "errors": 0}
    cycle_id = f"sim-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"

    for idea in ideas:
        idea_id_short = idea.get("id", "?")[:8]
        try:
            # Parse params from JSONB
            for param_key in ["power_params", "thermal_params", "data_movement_params", "pdn_params"]:
                if isinstance(idea.get(param_key), str):
                    import json
                    idea[param_key] = json.loads(idea[param_key])

            logger.info(f"[SimRunner] Simulating {idea_id_short} — '{idea.get('title','?')[:60]}'")
            report = route_idea(idea)

            # Push result via sim_feedback_loop (runs Inversion automatically)
            try:
                from db.sim_feedback_loop import push_sim_result_to_db, increment_iteration_count
                report_dict = report.model_dump(mode="json") if hasattr(report, "model_dump") else {}
                push_sim_result_to_db(sb, idea["id"], cycle_id, report_dict)

                # If near-miss: increment iteration counter
                if report.overall_score and 3.5 <= report.overall_score <= 6.5:
                    increment_iteration_count(sb, idea["id"])
                    results["near_miss"] += 1
            except ImportError:
                # Fallback: push directly to sim_results table
                import json as _json
                sb.table("sim_results").upsert({
                    "idea_id":      idea["id"],
                    "cycle_id":     cycle_id,
                    "sim_status":   report.overall_status,
                    "sim_score":    report.overall_score,
                    "recommendation": report.recommendation,
                    "revision_targets": _json.dumps(report.revision_targets),
                    "timestamp":    datetime.now(timezone.utc).isoformat(),
                }).execute()

            # Track stats
            status = report.overall_status
            if status in results:
                results[status] += 1
            else:
                results["marginal"] = results.get("marginal", 0) + 1

            logger.info(
                f"[SimRunner] {idea_id_short} → {status} "
                f"(score={report.overall_score:.1f}, rec={report.recommendation})"
            )

        except Exception as e:
            logger.error(f"[SimRunner] Failed for {idea_id_short}: {e}")
            results["errors"] += 1
            continue

    logger.info(
        f"[SimRunner] Complete: pass={results['pass']} fail={results['fail']} "
        f"near_miss={results['near_miss']} errors={results['errors']}"
    )
    logger.info(
        "[SimRunner] Results in Supabase: sim_results table + ideas.sim_score updated. "
        "HypothesisGenerator will read these in next Cycle 4."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea-id",   default="", help="Specific idea UUID")
    parser.add_argument("--force-all", default="false", help="Force all diamonds")
    args = parser.parse_args()
    run(idea_id=args.idea_id, force_all=args.force_all.lower() == "true")
