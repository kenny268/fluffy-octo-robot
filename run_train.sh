#!/usr/bin/env bash
# macOS: avoid OpenMP duplicate lib abort with PyTorch + NumPy (Conda).
export KMP_DUPLICATE_LIB_OK=TRUE

set -euo pipefail
cd "$(dirname "$0")"

python train.py "$@"
