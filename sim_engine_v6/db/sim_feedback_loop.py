"""
sim_feedback_loop.py — v7: Closed-Loop Sim Engine Integration

THE MISSING PIECE (was identified in analysis, now built):
  Sim Engine writes results to Supabase.
  This module reads them back and injects into:
    1. HypothesisGenerator  → replaces text kill_reason with NUMBERS
    2. PhysicsLimitMapper   → replaces JEDEC estimates with MEASURED limits
    3. ChiefScientist       → sim_score overrides LLM physics_feasibility guess
    4. Research Loop        → near-miss ideas get focused parameter-targeted searches

INVERSION ENGINE:
  For every near-miss (sim_score 4-6), computes:
    "What single parameter change turns this fail into a pass?"
  Example: R_theta=0.20 C/W failed. Critical=0.18.
    → revision_target: "R_theta must drop from 0.20 to 0.17 C/W (15% reduction needed)"
    → PhysicsLimitMapper gets: "engineering_limit: R_theta=0.18 C/W is the measured wall"
    → HypothesisGenerator gets: "15 ideas failed because R_theta > 0.18 — is this fixable?"

COST TRACKING:
  Every LLM call cost is logged to api_cost_log.
  Pricing table covers all 6 providers in RESEARCH_CHAIN + REASONING_CHAIN.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ── LLM pricing table (USD per 1M tokens, as of 2025) ────────────────────────
# Source: provider pricing pages. Update quarterly.
PRICING = {
    # (input_per_1m, output_per_1m)
    "groq":        {"llama-3.3-70b-versatile": (0.59, 0.79)},
    "sambanova":   {"DeepSeek-V3.2": (0.0, 0.0)},           # free tier
    "fireworks":   {"accounts/fireworks/models/deepseek-v3p2": (0.90, 2.70)},
    "mistral":     {"mistral-large-latest": (2.00, 6.00)},
    "gemini":      {"gemini-2.5-flash": (0.075, 0.30)},
    "cohere":      {"command-a-reasoning-08-2025": (2.50, 10.00)},
}

DEFAULT_COST_PER_1M = (1.00, 3.00)  # fallback if model not in table


def compute_cost_usd(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """Compute USD cost for a single LLM call."""
    p_in, p_out = PRICING.get(provider, {}).get(model, DEFAULT_COST_PER_1M)
    return round((tokens_in / 1e6) * p_in + (tokens_out / 1e6) * p_out, 6)


def log_api_cost(
    db_client,
    cycle_id: str,
    agent_name: str,
    llm_provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    duration_ms: int,
) -> bool:
    """Log a single LLM call cost to api_cost_log."""
    cost = compute_cost_usd(llm_provider, model, tokens_in, tokens_out)
    try:
        db_client.table("api_cost_log").insert({
            "cycle_id":    cycle_id,
            "agent_name":  agent_name,
            "llm_provider": llm_provider,
            "model":        model,
            "tokens_in":    tokens_in,
            "tokens_out":   tokens_out,
            "cost_usd":     cost,
            "duration_ms":  duration_ms,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.warning(f"[CostLog] Failed to log cost: {e}")
        return False


def get_cycle_cost(db_client, cycle_id: str) -> dict:
    """Get total cost for a cycle."""
    try:
        res = db_client.table("api_cost_log") \
            .select("cost_usd, tokens_in, tokens_out, llm_provider") \
            .eq("cycle_id", cycle_id) \
            .execute()
        rows = res.data or []
        return {
            "total_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "total_tokens": sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rows),
            "calls": len(rows),
            "by_provider": {
                p: round(sum(r["cost_usd"] for r in rows if r["llm_provider"] == p), 4)
                for p in set(r["llm_provider"] for r in rows)
            },
        }
    except Exception as e:
        logger.warning(f"[CostLog] get_cycle_cost failed: {e}")
        return {"total_usd": 0.0, "total_tokens": 0, "calls": 0, "by_provider": {}}


# ── Inversion Engine ──────────────────────────────────────────────────────────

def compute_revision_targets(sim_result: dict) -> List[dict]:
    """
    Inversion: given a sim result, compute what parameter changes would flip fail → pass.

    Returns list of revision targets:
      [{"parameter": "R_theta_c_per_w", "current": 0.20, "required": 0.17,
        "delta_pct": 15.0, "domain": "thermal",
        "description": "R_theta must drop from 0.20 to 0.17 C/W (15% reduction)"}]
    """
    targets = []

    # ── Thermal inversion ─────────────────────────────────────────────────────
    r_theta_actual   = sim_result.get("r_theta_actual")
    r_theta_critical = sim_result.get("r_theta_critical")
    if r_theta_actual and r_theta_critical and r_theta_actual > r_theta_critical:
        # Required: drop to 90% of critical (safety margin)
        required = round(r_theta_critical * 0.90, 4)
        delta_pct = round((r_theta_actual - required) / r_theta_actual * 100, 1)
        targets.append({
            "parameter":    "R_theta_c_per_w",
            "current_value": r_theta_actual,
            "required_value": required,
            "delta_pct":    delta_pct,
            "domain":       "thermal",
            "description":  (
                f"Thermal resistance must drop from {r_theta_actual:.3f} to {required:.3f} °C/W "
                f"({delta_pct:.0f}% reduction). "
                f"R_theta_critical={r_theta_critical:.4f} °C/W (from brentq solver). "
                f"Options: better TIM, direct bonding, microfluidic cooling."
            ),
        })

    # ── Temperature inversion ─────────────────────────────────────────────────
    t_op = sim_result.get("t_op_c")
    JEDEC_LIMIT = 125.0
    if t_op and t_op > JEDEC_LIMIT:
        excess = round(t_op - JEDEC_LIMIT, 1)
        # To bring T_op down by `excess` degrees, need R_theta * excess reduction
        targets.append({
            "parameter":    "t_op_c",
            "current_value": t_op,
            "required_value": JEDEC_LIMIT - 5.0,  # 5°C safety margin
            "delta_pct":    round(excess / t_op * 100, 1),
            "domain":       "thermal",
            "description":  (
                f"Operating temperature {t_op:.1f}°C exceeds JEDEC {JEDEC_LIMIT}°C by {excess:.1f}°C. "
                f"Need {excess + 5:.0f}°C reduction. "
                f"At current R_theta, requires {round((excess+5)*100/t_op, 0):.0f}% power reduction "
                f"OR equivalent thermal resistance improvement."
            ),
        })

    # ── Yield inversion ───────────────────────────────────────────────────────
    yield_pct = sim_result.get("yield_pct")
    if yield_pct is not None and yield_pct < 90.0:
        targets.append({
            "parameter":    "yield_pct",
            "current_value": yield_pct,
            "required_value": 95.0,
            "delta_pct":    round((95.0 - yield_pct) / 95.0 * 100, 1),
            "domain":       "yield",
            "description":  (
                f"Monte Carlo yield {yield_pct:.1f}% below 95% production target. "
                f"Top sensitivity: {sim_result.get('top_sensitivity_param', 'unknown')}. "
                f"Tighten that parameter's process variation to hit yield target."
            ),
        })

    # ── MTTF inversion ────────────────────────────────────────────────────────
    mttf = sim_result.get("min_mttf_years")
    if mttf is not None and mttf < 10.0:
        targets.append({
            "parameter":    "min_mttf_years",
            "current_value": mttf,
            "required_value": 10.0,
            "delta_pct":    round((10.0 - mttf) / 10.0 * 100, 1),
            "domain":       "reliability",
            "description":  (
                f"MTTF {mttf:.1f} years below JEDEC 10yr target. "
                f"Limiting mechanism: {sim_result.get('top_failure_domain', 'unknown')}. "
                f"Options: reduce T_op by 10°C → doubles MTTF for EM/HCI."
            ),
        })

    return targets


def is_near_miss(sim_score: float, revision_targets: List[dict]) -> bool:
    """
    Near-miss: sim_score 3.5-6.5 AND at least one revision target with delta < 25%.
    These are the most valuable ideas: close to viable, just need focused research.
    """
    if not (3.5 <= sim_score <= 6.5):
        return False
    if not revision_targets:
        return False
    return any(t.get("delta_pct", 100) < 25.0 for t in revision_targets)


# ── Sim Results → DB ─────────────────────────────────────────────────────────

def push_sim_result_to_db(
    db_client,
    idea_id: str,
    cycle_id: str,
    sim_report: dict,
) -> bool:
    """
    Write sim results to sim_results table AND update ideas table with
    sim_score, near_miss, revision_targets.

    sim_report: dict from Sim Engine's SimReport/SimResult.
    """
    try:
        # Extract key scalars
        overall_status = sim_report.get("overall_status", "skipped")
        sim_score      = float(sim_report.get("overall_score", sim_report.get("sim_score", 0.0)))
        recommendation = sim_report.get("recommendation", "")

        # Find key numbers across domain results
        r_theta_actual   = None
        r_theta_critical = None
        t_op_c           = None
        margin_pct       = None
        yield_pct        = None
        min_mttf_years   = None
        top_failure_domain = None

        domain_results = sim_report.get("domain_results", [])
        for dr in domain_results:
            domain = dr.get("domain", "")
            details = dr.get("details", {})
            if domain == "thermal":
                if details.get("coupled"):
                    coupled = details["coupled"]
                    r_theta_actual   = coupled.get("r_theta_actual") or r_theta_actual
                    r_theta_critical = coupled.get("r_theta_critical") or r_theta_critical
                    margin_pct       = coupled.get("margin_to_runaway_pct") or margin_pct
                    tt = coupled.get("tt_corner") or {}
                    t_op_c = tt.get("T_op_c") or t_op_c
            if domain == "yield":
                yield_pct = details.get("yield_pct") or yield_pct
            if domain == "reliability":
                min_mttf_years = details.get("min_mttf_years") or min_mttf_years
            if dr.get("status") in ("fail", "critical") and not top_failure_domain:
                top_failure_domain = domain

        # Build flat sim_result dict for inversion
        sim_flat = {
            "r_theta_actual":       r_theta_actual,
            "r_theta_critical":     r_theta_critical,
            "t_op_c":               t_op_c,
            "margin_to_runaway_pct": margin_pct,
            "yield_pct":            yield_pct,
            "min_mttf_years":       min_mttf_years,
            "top_failure_domain":   top_failure_domain,
            "top_sensitivity_param": sim_report.get("top_sensitivity_param"),
        }

        # Compute revision targets (inversion)
        revision_targets = compute_revision_targets(sim_flat)

        # Check near-miss
        near_miss = is_near_miss(sim_score, revision_targets)

        # Write to sim_results table
        db_client.table("sim_results").upsert({
            "idea_id":              idea_id,
            "cycle_id":             cycle_id,
            "sim_status":           overall_status,
            "sim_score":            sim_score,
            "near_miss":            near_miss,
            "recommendation":       recommendation,
            "r_theta_actual":       r_theta_actual,
            "r_theta_critical":     r_theta_critical,
            "t_op_c":               t_op_c,
            "margin_to_runaway_pct": margin_pct,
            "yield_pct":            yield_pct,
            "min_mttf_years":       min_mttf_years,
            "top_failure_domain":   top_failure_domain,
            "revision_targets":     json.dumps(revision_targets),
            "domain_results":       json.dumps(domain_results),
            "cross_domain_couplings": json.dumps(sim_report.get("cross_domain_couplings", [])),
            "duration_ms":          sim_report.get("duration_ms"),
            "timestamp":            datetime.now(timezone.utc).isoformat(),
        }).execute()

        # Update ideas table with sim feedback
        idea_update = {
            "sim_score":        sim_score,
            "near_miss":        near_miss,
            "revision_targets": json.dumps(revision_targets),
            "last_sim_at":      datetime.now(timezone.utc).isoformat(),
        }
        # If sim kills the idea, update status
        if recommendation == "kill_physics":
            kill_detail = "; ".join(
                t["description"] for t in revision_targets[:2]
            ) if revision_targets else overall_status
            idea_update["status"] = "killed"
            idea_update["kill_reason"] = f"[SimEngine-v7] {kill_detail[:400]}"
            idea_update["physics_kill_detail"] = json.dumps({
                "sim_score": sim_score,
                "top_failure_domain": top_failure_domain,
                "revision_targets": revision_targets[:3],
            })
        db_client.table("ideas").update(idea_update).eq("id", idea_id).execute()

        logger.info(
            f"[SimFeedback] idea={idea_id[:8]} | score={sim_score:.1f} | "
            f"near_miss={near_miss} | targets={len(revision_targets)} | "
            f"status={overall_status}"
        )
        return True

    except Exception as e:
        logger.error(f"[SimFeedback] push_sim_result_to_db failed for {idea_id}: {e}")
        return False


# ── Sim Results → HypothesisGenerator context ────────────────────────────────

def load_sim_kill_patterns(db_client, limit: int = 100) -> List[dict]:
    """
    Load sim failure patterns for HypothesisGenerator.

    Returns structured data:
      [{"parameter": "R_theta_c_per_w", "count": 23, "avg_actual": 0.21,
        "avg_critical": 0.17, "avg_delta_needed_pct": 19.0, "domain": "thermal"}]

    This replaces text kill_reason patterns with NUMBERS.
    """
    try:
        res = db_client.table("sim_results") \
            .select("top_failure_domain, r_theta_actual, r_theta_critical, t_op_c, "
                    "margin_to_runaway_pct, yield_pct, revision_targets, sim_score") \
            .in_("sim_status", ["fail", "critical"]) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        rows = res.data or []

        # Aggregate by failure domain
        from collections import defaultdict
        domain_stats: Dict[str, list] = defaultdict(list)
        for row in rows:
            domain = row.get("top_failure_domain") or "unknown"
            domain_stats[domain].append(row)

        patterns = []
        for domain, domain_rows in domain_stats.items():
            r_thetas = [r["r_theta_actual"] for r in domain_rows if r.get("r_theta_actual")]
            r_crits  = [r["r_theta_critical"] for r in domain_rows if r.get("r_theta_critical")]
            t_ops    = [r["t_op_c"] for r in domain_rows if r.get("t_op_c")]
            scores   = [r["sim_score"] for r in domain_rows if r.get("sim_score") is not None]

            pattern = {
                "domain":           domain,
                "failure_count":    len(domain_rows),
                "avg_sim_score":    round(sum(scores) / len(scores), 2) if scores else 0,
            }
            if r_thetas:
                pattern["avg_r_theta_actual"]   = round(sum(r_thetas) / len(r_thetas), 4)
            if r_crits:
                pattern["avg_r_theta_critical"] = round(sum(r_crits) / len(r_crits), 4)
                if r_thetas:
                    avg_excess = sum(a - c for a, c in zip(r_thetas, r_crits)) / len(r_thetas)
                    avg_delta  = avg_excess / (sum(r_thetas) / len(r_thetas)) * 100
                    pattern["avg_improvement_needed_pct"] = round(avg_delta, 1)
            if t_ops:
                pattern["avg_t_op_c"] = round(sum(t_ops) / len(t_ops), 1)

            patterns.append(pattern)

        patterns.sort(key=lambda x: x["failure_count"], reverse=True)
        return patterns

    except Exception as e:
        logger.warning(f"[SimFeedback] load_sim_kill_patterns failed: {e}")
        return []


def load_near_miss_ideas(db_client, limit: int = 20) -> List[dict]:
    """
    Load near-miss ideas with their revision targets.
    Used by Research Loop to generate focused searches.
    """
    try:
        res = db_client.table("ideas") \
            .select("id, title, domain, diamond_score, sim_score, revision_targets, "
                    "iteration_count, problem_statement, physical_limit") \
            .eq("near_miss", True) \
            .in_("status", ["active", "diamond"]) \
            .order("diamond_score", desc=True) \
            .limit(limit) \
            .execute()
        rows = res.data or []
        # Parse JSONB revision_targets if stored as string
        for row in rows:
            if isinstance(row.get("revision_targets"), str):
                try:
                    row["revision_targets"] = json.loads(row["revision_targets"])
                except Exception:
                    row["revision_targets"] = []
        return rows
    except Exception as e:
        logger.warning(f"[SimFeedback] load_near_miss_ideas failed: {e}")
        return []


def load_measured_physics_limits(db_client) -> List[dict]:
    """
    Load measured physics limits from sim_results.
    Feeds PhysicsLimitMapper as MEASURED limits (not JEDEC estimates).

    Returns:
      [{"domain": "thermal", "parameter": "R_theta_c_per_w",
        "measured_limit": 0.18, "n_datapoints": 23,
        "description": "R_theta_critical measured at 0.18 C/W across 23 sim runs"}]
    """
    try:
        res = db_client.table("sim_results") \
            .select("top_failure_domain, r_theta_critical, t_op_c, yield_pct, min_mttf_years") \
            .not_.is_("r_theta_critical", "null") \
            .order("created_at", desc=True) \
            .limit(200) \
            .execute()
        rows = res.data or []

        if not rows:
            return []

        r_crits = [r["r_theta_critical"] for r in rows if r.get("r_theta_critical")]
        measured = []
        if r_crits:
            avg_crit = round(sum(r_crits) / len(r_crits), 4)
            min_crit = round(min(r_crits), 4)
            measured.append({
                "domain":          "thermal",
                "parameter":       "R_theta_c_per_w",
                "measured_limit":  avg_crit,
                "tightest_limit":  min_crit,
                "n_datapoints":    len(r_crits),
                "limit_type":      "engineering",
                "description": (
                    f"R_theta_critical measured at avg={avg_crit:.4f} C/W "
                    f"(tightest={min_crit:.4f} C/W) across {len(r_crits)} sim runs. "
                    f"This is the MEASURED thermal runaway boundary, not JEDEC estimate."
                ),
            })

        return measured

    except Exception as e:
        logger.warning(f"[SimFeedback] load_measured_physics_limits failed: {e}")
        return []


def increment_iteration_count(db_client, idea_id: str) -> bool:
    """Increment the Research Loop iteration counter for an idea."""
    try:
        res = db_client.table("ideas").select("iteration_count").eq("id", idea_id).execute()
        current = (res.data[0].get("iteration_count") or 0) if res.data else 0
        db_client.table("ideas").update({"iteration_count": current + 1}).eq("id", idea_id).execute()
        return True
    except Exception as e:
        logger.warning(f"[SimFeedback] increment_iteration_count failed: {e}")
        return False
