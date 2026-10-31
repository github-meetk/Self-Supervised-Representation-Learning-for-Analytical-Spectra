#!/usr/bin/env bash
set -euo pipefail
spectra-rl train --config "${1:-configs/default.yaml}"
spectra-rl evaluate --config "${1:-configs/default.yaml}"
spectra-rl report --config "${1:-configs/default.yaml}"
