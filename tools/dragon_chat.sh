#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dragon_architecture=llama
dragon_default_upstream="$HOME/.cache/tinyml-llama2c"
dragon_default_checkpoint="$project_dir/build/dragon/current.pt"
if [[ "${1:-}" == tinytalk ]]; then
  shift
  dragon_architecture=gpt-neo
  dragon_default_upstream="$HOME/esp32-tinytalk/data/chat_model_8m"
  dragon_default_checkpoint="$project_dir/build/dragon/neo-training/best.pt"
elif [[ "${1:-}" == stories ]]; then
  shift
fi
dragon_python="${DRAGON_PYTHON:-$HOME/.cache/tinyml-eval-venv/bin/python}"
dragon_upstream="${DRAGON_UPSTREAM:-$dragon_default_upstream}"
dragon_checkpoint="${DRAGON_CHECKPOINT:-$dragon_default_checkpoint}"
if [[ ! -x "$dragon_python" || ! -f "$dragon_checkpoint" ]]; then
  echo "Student environment or checkpoint missing. See docs/dragon-training.md."
  exit 1
fi
exec "$dragon_python" "$project_dir/tools/dragon_student.py" chat \
  --architecture "$dragon_architecture" --upstream "$dragon_upstream" --checkpoint "$dragon_checkpoint" "$@"
