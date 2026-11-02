#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
if [ -d "$PROJECT_DIR/.tools/jdk17/Contents/Home" ]; then
    export JAVA_HOME="$PROJECT_DIR/.tools/jdk17/Contents/Home"
fi
export SPARK_LOCAL_IP=127.0.0.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
S4_SCRIPT=preprocessing/week4.py
if [ "${1:-}" = "benchmark" ]; then
    shift
    S4_SCRIPT=preprocessing/benchmark_s4.py
fi
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$PROJECT_DIR/.venv-spark/bin/python" "$S4_SCRIPT" "$@"
fi
exec "$PROJECT_DIR/.venv-spark/bin/python" "$S4_SCRIPT" "$@"
