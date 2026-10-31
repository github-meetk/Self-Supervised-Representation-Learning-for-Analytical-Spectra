#!/usr/bin/env bash
set -euo pipefail
spectra-rl evaluate --config "${1:-configs/default.yaml}"
