"""
reader.py — Reads Diamond ideas from RD Engine's Supabase.
Also writes SimResult back to a new sim_results table.
"""
from __future__ import annotations
import json, logging, os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_client = None

def get_client():
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def load_diamond_ideas(limit: int = 20) -> List[dict]:
    """Load Diamond ideas with extracted params from Supabase."""
    try:
        db = get_client()
        res = db.table("ideas") \
            .select("id,title,domain,problem_statement,physical_limit,diamond_score,"
                    "power_params,thermal_params,data_movement_params,pdn_params,"
                    "company_context,proposed_direction") \
            .eq("status", "diamond") \
            .order("diamond_score", desc=True) \
            .limit(limit) \
            .execute()
        ideas = res.data or []
        logger.info(f"[DB] Loaded {len(ideas)} Diamond ideas")

        # Parse JSONB fields
        for idea in ideas:
            for field in ["power_params","thermal_params","data_movement_params","pdn_params"]:
                val = idea.get(field)
                if isinstance(val, str):
                    try:
                        idea[field] = json.loads(val)
                    except Exception:
                        idea[field] = {}
                elif val is None:
                    idea[field] = {}
        return ideas
    except Exception as e:
        logger.error(f"[DB] load_diamond_ideas failed: {e}")
        return []


def load_active_high_score_ideas(min_score: float = 7.0, limit: int = 30) -> List[dict]:
    """Load high-scoring active ideas (not just diamonds) for simulation."""
    try:
        db = get_client()
        res = db.table("ideas") \
            .select("id,title,domain,problem_statement,diamond_score,"
                    "power_params,thermal_params,data_movement_params,pdn_params") \
            .in_("status", ["diamond", "active"]) \
            .gte("diamond_score", min_score) \
            .order("diamond_score", desc=True) \
            .limit(limit) \
            .execute()
        ideas = res.data or []
        for idea in ideas:
            for field in ["power_params","thermal_params","data_movement_params","pdn_params"]:
                val = idea.get(field)
                if isinstance(val, str):
                    try:
                        idea[field] = json.loads(val)
                    except Exception:
                        idea[field] = {}
                elif val is None:
                    idea[field] = {}
        return ideas
    except Exception as e:
        logger.error(f"[DB] load_active failed: {e}")
        return []


def save_sim_result(sim_result, idea_id: str) -> bool:
    """Save SimResult to sim_results table."""
    try:
        db = get_client()
        row = {
            "sim_id":           sim_result.sim_id,
            "idea_id":          idea_id,
            "idea_title":       sim_result.idea_title,
            "overall_status":   sim_result.overall_status,
            "sim_score":        sim_result.sim_score,
            "critical_failures": sim_result.critical_failures,
            "warnings":         sim_result.warnings,
            "key_insights":     sim_result.key_insights,
            "thermal_result":   sim_result.thermal.model_dump(mode="json") if sim_result.thermal else None,
            "pdn_result":       sim_result.pdn.model_dump(mode="json")     if sim_result.pdn     else None,
            "electrical_result": sim_result.electrical.model_dump(mode="json") if sim_result.electrical else None,
            "dm_result":        sim_result.data_movement.model_dump(mode="json") if sim_result.data_movement else None,
            "duration_ms":      sim_result.duration_ms,
            "report_path":      sim_result.report_path,
        }
        db.table("sim_results").upsert(row).execute()
        logger.info(f"[DB] Saved SimResult for {idea_id[:8]}")
        return True
    except Exception as e:
        logger.error(f"[DB] save_sim_result failed: {e}")
        return False


def get_sim_schema_sql() -> str:
    return """
-- sim_results table — run once in Supabase SQL editor
CREATE TABLE IF NOT EXISTS sim_results (
    sim_id          UUID PRIMARY KEY,
    idea_id         UUID REFERENCES ideas(id) ON DELETE CASCADE,
    idea_title      TEXT,
    overall_status  TEXT CHECK (status IN ('pass','warning','critical','fail','skipped')),
    sim_score       FLOAT DEFAULT 0.0,
    critical_failures JSONB DEFAULT '[]',
    warnings        JSONB DEFAULT '[]',
    key_insights    JSONB DEFAULT '[]',
    thermal_result  JSONB,
    pdn_result      JSONB,
    electrical_result JSONB,
    dm_result       JSONB,
    duration_ms     INTEGER,
    report_path     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sim_results_idea  ON sim_results(idea_id);
CREATE INDEX IF NOT EXISTS idx_sim_results_score ON sim_results(sim_score DESC);
"""
