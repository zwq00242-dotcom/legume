"""Repository-relative locations for the independent ACS diagnosis."""
from pathlib import Path
import os

BASE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
CACHE = BASE / '.cache'
for directory in [CACHE, CACHE / 'matplotlib', BASE / 'data', BASE / 'figures', BASE / 'validation', BASE / 'reports']:
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault('XDG_CACHE_HOME', str(CACHE))
