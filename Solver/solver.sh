#!/bin/bash
# (c)2014 Federico Cerutti <federico.cerutti@acm.org> --- MIT LICENCE
# adapted 2016 by Thomas Linsbichler <linsbich@dbai.tuwien.ac.at> --- MIT LICENSE
# adapted 2018 by Francesco Santini <francesco.santini@dmi.unipg.it> --- MIT LICENSE
# adapted 2018 by Theofrastos Mantadelis <theo.mantadelis@dmi.unipg.it> --- MIT LICENSE
# Generic script interface for ICCMA 2019

# Quoted ICCMA-style adapter for the Python solver's supported input format.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fail() { printf '%s\n' "$*" >&2; exit 2; }
if [[ $# -eq 0 ]]; then
    printf 'AFGCN\nLars Malmqvist\n'
    exit 0
fi
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --formats) printf '[i23]\n'; exit 0 ;;
        --problems)
            printf '[DS-CO, DC-CO, DS-PR, DC-PR, DS-ST, DC-ST, DS-SST, DC-SST, DS-STG, DC-STG, DS-ID]\n'
            exit 0 ;;
        -h|--help)
            printf 'Usage: solver.sh -p TASK -f FILE -a ARGUMENT [-fo i23] [--seed N]\n'
            exit 0 ;;
        -p|-f|-a|-fo|--seed)
            [[ $# -ge 2 ]] || fail "Missing value for $1"
            case "$1" in
                -p) args+=(--task "$2") ;;
                -f) args+=(--filepath "$2") ;;
                -a) args+=(--argument "$2") ;;
                --seed) args+=(--seed "$2") ;;
                -fo) [[ "$2" == i23 ]] || fail 'Supported format: i23 (p af N followed by attack pairs)' ;;
            esac
            shift 2 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
exec "${PYTHON:-python}" "$SCRIPT_DIR/solver.py" "${args[@]}"
