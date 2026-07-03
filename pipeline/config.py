"""Central paths and settings for the G1 dance pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
THIRD_PARTY = PROJECT_ROOT / "third_party"
LOGS_DIR = PROJECT_ROOT / "logs"

# Robot constants (see ~/robot/RUNBOOK.md)
ROBOT_PC2_IP = "192.168.123.164"
LAPTOP_WIRED_IP = "192.168.123.2"
ROBOT_NET_IFACE_CONNECTION = "robot-lan"

# Stage names in execution order. Implementations live in pipeline/stages/.
STAGE_ORDER = ["extract", "retarget", "train", "verify", "export"]

# Performance venues (see pipeline/venue.py). The spatial gate is per-venue, not a
# hardcoded constant: max root excursion = venue radius minus a safety margin.
VENUES_DIR = DATA_DIR / "venues"
# Env var the app sets when running vet_motion.py / find_window.py as subprocesses
# to apply the selected venue's max excursion. Absent => 1.5 m (default "Home" venue).
VENUE_EXCURSION_ENV = "G1_MAX_EXCURSION_M"
DEFAULT_MAX_EXCURSION_M = 1.5

JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
