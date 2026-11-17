#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
if [ -d "$PROJECT_DIR/.tools/jdk17/Contents/Home" ]; then
    export JAVA_HOME="$PROJECT_DIR/.tools/jdk17/Contents/Home"
fi
export SPARK_LOCAL_IP=127.0.0.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec caffeinate -i "$PROJECT_DIR/.venv-spark/bin/python" analysis/post_stats.py "$@"
