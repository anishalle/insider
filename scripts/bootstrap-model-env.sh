#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
require_torch="${2:-0}"

cd "$repo_root"

python - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "duckdb": "duckdb>=1.2.2",
    "numpy": "numpy>=2.0.0",
    "pyarrow": "pyarrow>=18.0.0",
    "sklearn": "scikit-learn>=1.5.0",
    "xgboost": "xgboost>=2.1.0",
}
if sys.version_info < (3, 11):
    required["tomli"] = "tomli>=2.0.1"
missing = [package for module_name, package in required.items() if importlib.util.find_spec(module_name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

if [[ "$require_torch" == "1" ]]; then
  python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    sys.stderr.write("torch is not available in the active environment.\n")
    sys.exit(1)
PY
fi
