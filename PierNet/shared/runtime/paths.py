import os
from pathlib import Path

from PierNet.shared.runtime.env import load_env_file

load_env_file()


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, '').strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


PROJECT_ROOT = _env_path('PierNet_ROOT', Path(__file__).resolve().parents[3])
DATA_ROOT = _env_path('PierNet_DATA_ROOT', PROJECT_ROOT / 'data')
ARTIFACT_ROOT = _env_path('PierNet_ARTIFACT_ROOT', PROJECT_ROOT / 'artifacts')
RUNLOG_ROOT = _env_path('PierNet_RUNLOG_ROOT', PROJECT_ROOT / '.runlogs')
DATA_DIR = DATA_ROOT / 'text2comp'
TEMPLATES_DIR = DATA_ROOT / 'templates'
CONFIGS_ROOT = PROJECT_ROOT / 'configs'
CONFIG_DIR = CONFIGS_ROOT / 'text2comp'
REGISTRY_PATH = CONFIG_DIR / 'registry.yaml'
