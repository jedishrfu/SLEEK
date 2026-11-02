#!/usr/bin/env bash
# Run one sweep per file and collect all rows in a fresh results CSV.
# Usage: ./runtests.sh [input-directory] [additional sweep options]
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input_dir="${1:-/home/LC/SDRbench_inputs/ISABEL/100x500x500/}"
if (( $# > 0 )); then shift; fi
if [[ ! -d "$input_dir" ]]; then
    echo "Input directory does not exist: $input_dir" >&2
    exit 1
fi
# Protect the combined report from conflicting output options.
for arg in "$@"; do
    case "$arg" in
        -o|--output|--output=*|-o?*|--append)
            echo "runtests.sh manages --output and --append automatically" >&2
            exit 1 ;;
    esac
done
shopt -s nullglob dotglob
files=()
for input in "$input_dir"/*; do
    [[ -f "$input" ]] && files+=("$input")
done
if (( ${#files[@]} == 0 )); then
    echo "No input files found in: $input_dir" >&2
    exit 1
fi
output="$script_dir/sleek_results.csv"
: > "$output"
for input in "${files[@]}"; do
    echo "Running sweep: $input"
    python3 "$script_dir/run_sleek_sweep.py" "$input" --append --output "$output" "$@"
done
echo "Completed ${#files[@]} files. Results: $output"
