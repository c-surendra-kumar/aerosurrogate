import sys
from pathlib import Path

# Tests import the project's flat modules (naca, features, api, ...), so the
# repo root has to be importable regardless of where pytest was invoked from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
