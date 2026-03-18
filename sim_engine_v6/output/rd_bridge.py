"""
rd_bridge.py v7 — Integration bridge between rd_engine and sim_engine.

WHAT'S NEW IN V7:
  - push_sim_results_to_supabase now calls sim_feedback_loop.push_sim_result_to_db
    which runs the Inversion Engine and computes revision_targets AUTOMATICALLY.
  - near_miss detection is computed in sim_feedback_loop, not here.
  - Cost tracking: sim pipeline duration is logged.

FLOW:
  1. rd_bridge loads diamond ideas from Supabase
  2. route_idea() simulates each one
  3. push_sim_result_to_db() computes revision_targets + near_miss and writes to:
       a. sim_results table (quantitative data)
       b. ideas table (sim_score, near_miss, revision_targets)
  4. Cycle 4: HypothesisGenerator reads sim_results via load_sim_kill_patterns()
"""
from __future__ import annotations
import json
import logging
import sys, os
# Package root = sim_engine_v6/ (parent of output/)
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
from pathlib import Path
from typing import List, Dict, Any, Optional
from orchestrator.sim_router import route_idea
from core.schemas import BatchSimResult, SimReport
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def load_ideas_from_json(path: str) -> List[Dict[str, Any]]:
    """Load ideas from a JSON file (rd_engine export)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if "ideas" in data:
        return data["ideas"]
    return [data]


def load_diamonds_from_supabase(supabase_url: str, supabase_key: str) -> List[Dict[str, Any]]:
    """Pull diamond ideas from rd_engine's Supabase DB."""
    try:
        from supabase import create_client
        sb = create_client(supabase_url, supabase_key)
        response = sb.table("ideas").select("*").in_("status", ["diamond"]).execute()
        return response.data or []
    except ImportError:
        raise ImportError("pip install supabase to use Supabase integration")
    except Exception as e:
        raise RuntimeError(f"Supabase load failed: {e}")


def push_sim_results_to_supabase(
    reports: List[SimReport],
    supabase_url: str,
    supabase_key: str,
    cycle_id: Optional[str] = None,
) -> None:
    """
    Write sim results back to rd_engine's DB via sim_feedback_loop (v7).
    Automatically computes revision_targets and near_miss via Inversion Engine.
    """
    try:
        from supabase import create_client
        sb = create_client(supabase_url, supabase_key)

        # v7: use sim_feedback_loop (local copy of inversion engine — no cross-repo dependency)
        try:
            from db.sim_feedback_loop import push_sim_result_to_db
            use_feedback_loop = True
        except ImportError:
            use_feedback_loop = False
            logger.warning("[Bridge] sim_feedback_loop not found — falling back to legacy push")

        for report in reports:
            if use_feedback_loop:
                # v7 path: inversion engine runs automatically
                report_dict = report.model_dump(mode="json") if hasattr(report, 'model_dump') else dict(report)
                push_sim_result_to_db(sb, report.idea_id, cycle_id or "", report_dict)
            else:
                # legacy path: write directly (no inversion)
                sim_data = {
                    "idea_id":         report.idea_id,
                    "sim_status":      report.overall_status,
                    "sim_score":       report.overall_score,
                    "recommendation":  report.recommendation,
                    "revision_targets": json.dumps(report.revision_targets),
                    "domain_results":  json.dumps([
                        r.model_dump(mode="json") if hasattr(r, 'model_dump') else r
                        for r in report.domain_results
                    ]),
                    "cross_domain_couplings": json.dumps(report.cross_domain_couplings),
                    "timestamp": report.timestamp or datetime.now(timezone.utc).isoformat(),
                }
                sb.table("sim_results").upsert(sim_data).execute()
                if report.recommendation == "kill_physics":
                    kill_reason = "; ".join(report.revision_targets[:2])
                    sb.table("ideas").update({
                        "status":     "killed",
                        "kill_reason": f"[SimEngine] {kill_reason[:300]}",
                    }).eq("id", report.idea_id).execute()

        logger.info(f"[Bridge] Pushed {len(reports)} sim results to Supabase")

    except ImportError:
        raise ImportError("pip install supabase to use Supabase integration")


def run_sim_pipeline(
    ideas: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    push_to_db: bool = False,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    cycle_id: Optional[str] = None,
) -> BatchSimResult:
    """Full pipeline: ideas → simulate → save → optionally push to DB."""
    reports = []
    for idea in ideas:
        report = route_idea(idea)
        reports.append(report)

    summary = {
        "total":               len(reports),
        "pass":                sum(1 for r in reports if r.overall_status == "pass"),
        "fail":                sum(1 for r in reports if r.overall_status == "fail"),
        "marginal":            sum(1 for r in reports if r.overall_status == "marginal"),
        "proceed_to_prototype": sum(1 for r in reports if r.recommendation == "proceed_to_prototype"),
        "kill_physics":        sum(1 for r in reports if r.recommendation == "kill_physics"),
        "avg_score":           round(sum(r.overall_score for r in reports) / max(len(reports), 1), 2),
    }

    batch = BatchSimResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ideas_submitted=len(ideas),
        ideas_completed=len(reports),
        reports=reports,
        summary=summary,
    )

    if output_path:
        Path(output_path).write_text(batch.model_dump_json(indent=2))

    if push_to_db and supabase_url and supabase_key:
        push_sim_results_to_supabase(reports, supabase_url, supabase_key, cycle_id)

    return batch
