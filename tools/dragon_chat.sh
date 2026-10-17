#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dragon_python="${DRAGON_PYTHON:-$HOME/.cache/tinyml-eval-venv/bin/python}"
if [[ "${1:-}" == dialogue || "${1:-}" == dialogue-plain ]]; then
  dragon_variant="$1"
  shift
  dragon_model="$project_dir/build/dragon/dialogue-v1-retained/best"
  if [[ "$dragon_variant" == dialogue-plain ]]; then
    dragon_model="$project_dir/build/dragon/dialogue-v1/best"
  fi
  if [[ ! -x "$dragon_python" || ! -f "$dragon_model/model.safetensors" ]]; then
    echo "Trained checkpoint missing. See docs/dialogue-training-v1.md."
    exit 1
  fi
  exec "$dragon_python" "$project_dir/tools/dragon_hf.py" chat \
    --model "$dragon_model" --persona compact --context 256 "$@"
fi
if [[ "${1:-}" == minimind || "${1:-}" == smollm ]]; then
  dragon_candidate="$1"
  shift
  if [[ "$dragon_candidate" == minimind ]]; then
    dragon_model="$HOME/.cache/tinyml-minimind2-small"
    dragon_revision=8c0c0de640cd532fee03d329b95a58c56591b5bc
  else
    dragon_model="$HOME/.cache/tinyml-smollm2-135m-instruct"
    dragon_revision=12fd25f77366fa6b3b4b768ec3050bf629380bac
  fi
  if [[ ! -x "$dragon_python" || ! -f "$dragon_model/model.safetensors" ]]; then
    echo "Candidate environment or weights missing. See docs/dialogue-candidates.md."
    exit 1
  fi
  exec "$dragon_python" "$project_dir/tools/dragon_hf.py" chat \
    --model "$dragon_model" --source-revision "$dragon_revision" "$@"
fi
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
dragon_upstream="${DRAGON_UPSTREAM:-$dragon_default_upstream}"
dragon_checkpoint="${DRAGON_CHECKPOINT:-$dragon_default_checkpoint}"
if [[ ! -x "$dragon_python" || ! -f "$dragon_checkpoint" ]]; then
  echo "Student environment or checkpoint missing. See docs/dragon-training.md."
  exit 1
fi
exec "$dragon_python" "$project_dir/tools/dragon_student.py" chat \
  --architecture "$dragon_architecture" --upstream "$dragon_upstream" --checkpoint "$dragon_checkpoint" "$@"
