from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / 'data' / 'text2comp'
TEMPLATES_DIR = PROJECT_ROOT / 'data' / 'templates'
CONFIG_DIR = PROJECT_ROOT / 'configs' / 'text2comp'
CONFIGS_ROOT = PROJECT_ROOT / 'configs'
REGISTRY_PATH = CONFIG_DIR / 'registry.yaml'
