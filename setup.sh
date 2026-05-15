#!/bin/bash
set -e

NAME=${1:-"marknull"}
PY=${2:-"3.11"}
REQ=${3:-"requirements.txt"}

command -v conda &>/dev/null || { echo "❌ conda not found"; exit 1; }

if conda env list | grep -q "^${NAME} "; then
    read -p "⚠️  '$NAME' already exists. Recreate? (y/N): " c
    [[ "$c" =~ ^[Yy]$ ]] && conda env remove -n "$NAME" -y || { conda activate "$NAME"; [ -f "$REQ" ] && pip install -r "$REQ"; exit 0; }
fi

conda create -n "$NAME" python="$PY" -y
eval "$(conda shell.bash hook)" && conda activate "$NAME"
pip install -q --upgrade pip
[ -f "$REQ" ] && pip install -r "$REQ" || echo "⚠️  $REQ not found, skipping"

echo "✅ Done! Activate with: conda activate $NAME"