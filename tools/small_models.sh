#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SMALL_MODELS_PYTHON:-}" ]]; then
  PYTHON="$SMALL_MODELS_PYTHON"
elif [[ -x "$ROOT/build/model-playground/venv/bin/python" ]]; then
  PYTHON="$ROOT/build/model-playground/venv/bin/python"
elif [[ -x "$HOME/.cache/tinyml-eval-venv/bin/python" ]]; then
  PYTHON="$HOME/.cache/tinyml-eval-venv/bin/python"
else
  PYTHON=python3
fi
MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  printf '1) Stentor 30M — Q4, диалог\n2) Stories42M — Q4, продолжение истории\n3) VaghLM 42.6M — FP32, доступный checkpoint\n'
  read -r -p 'Выберите 1, 2 или 3: ' MODEL
else
  shift
fi
case "$MODEL" in
  1|stentor) MODEL=stentor ;;
  2|stories) MODEL=stories ;;
  3|vagh) MODEL=vagh ;;
  *) printf 'Использование: bash tools/small_models.sh [stentor|stories|vagh] [параметры]\n' >&2; exit 2 ;;
esac
exec "$PYTHON" "$ROOT/tools/small_models.py" chat "$MODEL" "$@"
