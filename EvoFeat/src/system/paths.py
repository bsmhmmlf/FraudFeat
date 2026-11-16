from __future__ import annotations

import sysconfig
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = SYSTEM_ROOT.parent
PROJECT_ROOT = SOURCE_ROOT.parent if SOURCE_ROOT.name == "src" else Path.cwd().resolve()
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_DIR = DATA_DIR / "memory"
PLAYBOOK_DIR = MEMORY_DIR / "skills"
PLAYBOOK_FILES = ("graph_paradigm.md", "explore_navigate.md", "feature_hypothesize.md",
                  "synthesis_playbook.md", "eval_calibration.md")
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"
if not (CONFIG_DIR / "fraudfeat_schema.json").is_file():
    CONFIG_DIR = (Path(sysconfig.get_path("data")) / "share"
                  / "evofeat" / "config")
