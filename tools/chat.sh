#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$project_dir/build/host"
mkdir -p "$build_dir"
"${CXX:-c++}" -std=c++17 -O2 -pthread \
    -I "$project_dir/tools/host" \
    -I "$project_dir/components/tinytalk/include" \
    "$project_dir/main/console.cpp" \
    "$project_dir/components/tinytalk/llm.cpp" \
    -o "$build_dir/tinyml-chat"

# The console implements its own echo and line editing for USB. Turn off the
# terminal's echo so local typing appears once; keep normal Ctrl-C handling.
if [[ -t 0 ]]; then
    terminal_state="$(stty -g)"
    trap 'stty "$terminal_state"' EXIT
    stty -icanon -echo min 1 time 0
fi
"$build_dir/tinyml-chat" \
    "$project_dir/components/tinytalk/assets/model_neo_q4.bin" \
    "$project_dir/components/tinytalk/assets/tok_neo.bin"
