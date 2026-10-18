#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_dir/build/host"
mkdir -p "$build_dir"
python3 "$project_dir/tools/prepare_esp32ai.py" --out "$build_dir/esp32ai"
"${CXX:-c++}" -std=c++17 -O3 \
    -I "$project_dir/components/esp32ai/upstream" \
    -I "$build_dir/esp32ai" \
    "$project_dir/main/console.cpp" \
    -o "$build_dir/tinyml-chat"

# The console implements its own echo and line editing for USB. Turn off the
# terminal's echo so local typing appears once; keep normal Ctrl-C handling.
if [[ -t 0 ]]; then
    terminal_state="$(stty -g)"
    trap 'stty "$terminal_state"' EXIT
    stty -icanon -echo min 1 time 0
fi
"$build_dir/tinyml-chat" "$project_dir/components/esp32ai/assets/model.bin"
