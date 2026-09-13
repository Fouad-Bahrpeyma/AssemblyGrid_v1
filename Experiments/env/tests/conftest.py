"""Make the benchmark modules importable for every test, independent of collection order."""
import sys
from pathlib import Path

_ENV = Path(__file__).resolve().parents[1]
if str(_ENV) not in sys.path:
    sys.path.insert(0, str(_ENV))
