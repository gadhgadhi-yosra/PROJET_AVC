#!/bin/sh
set -eu
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_DIR"
if [ -d "$PROJECT_DIR/.tools/jdk17/Contents/Home" ]; then
    export JAVA_HOME="$PROJECT_DIR/.tools/jdk17/Contents/Home"
else
    JAVA_HOME=$(/usr/libexec/java_home -v 17)
    export JAVA_HOME
fi
export PATH="$JAVA_HOME/bin:$PATH"
export SPARK_LOCAL_IP=127.0.0.1
export OMP_NUM_THREADS=1
# Spark perd ses heartbeats si le Mac se met en veille : caffeinate l'en empêche.
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$PROJECT_DIR/.venv-spark/bin/python" preprocessing/pipeline.py "$@"
fi
exec "$PROJECT_DIR/.venv-spark/bin/python" preprocessing/pipeline.py "$@"
