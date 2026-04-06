#!/usr/bin/env bash
set -euo pipefail

cd "${1:-$(pwd)}"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
