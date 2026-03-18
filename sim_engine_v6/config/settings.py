import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Simulation thresholds
MIN_DIAMOND_SCORE_TO_SIM = 7.0   # only simulate ideas scoring above this
SIM_REPORT_OUTPUT_DIR    = "reports/output"
SIM_PLOT_DPI             = 120
