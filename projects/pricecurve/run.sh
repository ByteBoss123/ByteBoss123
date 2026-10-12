#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash data/download.sh
python src/prepare_data.py
python src/model_bayesian.py
python src/optimizer.py
python src/plot.py
