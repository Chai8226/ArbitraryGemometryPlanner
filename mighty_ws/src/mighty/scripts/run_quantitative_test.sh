#!/usr/bin/env bash
set -eo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/noetic_env.sh" /usr/bin/python3 "$script_dir/offline_quantitative_test.py" "$@"
