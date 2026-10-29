#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <train_script> --config <config_file> [train_args...]

Launcher options (handled by this script):
  -e  Rendezvous endpoint, e.g. master:29500
        - Optional if -N 1 (defaults to localhost:0)
        - Required when -N > 1
  -R  Node rank for static rendezvous only           [default: \$NODE_RANK]
  -N  Number of nodes (nnodes)                     [default: 1 or \$NNODES]
  -P  Processes per node (GPUs/node)               [default: 1 or \$NPROC_PER_NODE]
  -t  OMP_NUM_THREADS                              [default: auto from cores/GPUs, or \$OMP_NUM_THREADS]
  -d  Dry run (print command and exit)
  -h  Show this help message

Set a shared, unique RDZV_ID for c10d multi-node launches.

Training script:
  <train_script>          Python training script to run (e.g., train_subtyping.py)
  --config <config_file>  Required; must appear immediately after <train_script>
  [train_args...]         Any additional args for <train_script>

Example (single node, default rendezvous = localhost:0):

  $(basename "$0") -N 1 -P 1 -t 4 \\
      train_subtyping.py --config configs/baseline.yaml --experiment baseline
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

# --------------------------------------------------------------------------- #
# Defaults (overridable via env or flags)
# --------------------------------------------------------------------------- #

nnodes="${NNODES:-1}"
nproc_per_node="${NPROC_PER_NODE:-1}"
rdzv_endpoint="${RDZV_ENDPOINT:-}"
rdzv_backend="${RDZV_BACKEND:-c10d}"
rdzv_id="${RDZV_ID:-}"
node_rank="${NODE_RANK:-}"

# If user set OMP_NUM_THREADS in env, respect it; else we'll auto-compute later
OMP_NUM_THREADS_VAL="${OMP_NUM_THREADS:-}"
dry_run=0

# --------------------------------------------------------------------------- #
# Parse launcher options (built-in getopts)
# --------------------------------------------------------------------------- #

while getopts ":N:P:R:e:t:dh" opt; do
    case "$opt" in
        N) nnodes="$OPTARG" ;;
        P) nproc_per_node="$OPTARG" ;;
        R) node_rank="$OPTARG" ;;
        e) rdzv_endpoint="$OPTARG" ;;
        t) OMP_NUM_THREADS_VAL="$OPTARG" ;;
        d) dry_run=1 ;;
        h) usage; exit 0 ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            exit 1
            ;;
        \?)
            echo "Unknown option: -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))

# --------------------------------------------------------------------------- #
# Positional: <train_script> --config <cfg> [train_args...]
# --------------------------------------------------------------------------- #

# Need at least: train_script, --config, config_file
if [ "$#" -lt 3 ]; then
    echo "Error: expected <train_script> --config <config_file> [train_args...]." >&2
    echo
    usage
    exit 1
fi

train_script="$1"
shift

if [ "$1" != "--config" ]; then
    die "first argument after <train_script> must be '--config'. Got '$1'."
fi

config_file="$2"
shift 2

train_args=(--config "$config_file" "$@")

# --------------------------------------------------------------------------- #
# Validation (nnodes / nproc / rendezvous)
# --------------------------------------------------------------------------- #

if ! [[ "$nnodes" =~ ^[1-9][0-9]*$ ]]; then
    die "-N <nnodes> must be a positive integer (got '$nnodes')."
fi

if ! [[ "$nproc_per_node" =~ ^[1-9][0-9]*$ ]]; then
    die "-P <nproc_per_node> must be a positive integer (got '$nproc_per_node')."
fi

if [ "$rdzv_backend" = "static" ]; then
    if [ "$nnodes" -gt 1 ] && [ -z "$node_rank" ]; then
        die "-R <node_rank> or NODE_RANK is required for static multi-node torchrun."
    fi
    node_rank="${node_rank:-0}"
    if ! [[ "$node_rank" =~ ^[0-9]+$ ]]; then
        die "-R <node_rank> must be a non-negative integer (got '$node_rank')."
    fi
    if [ "$node_rank" -ge "$nnodes" ]; then
        die "-R <node_rank> must be less than nnodes (got '$node_rank' for nnodes=$nnodes)."
    fi
elif [ -n "$node_rank" ]; then
    die "-R <node_rank> and NODE_RANK are valid only with static rendezvous."
elif [ "$nnodes" -gt 1 ] && [ -z "$rdzv_id" ]; then
    die "RDZV_ID is required for non-static multi-node torchrun."
fi

# If rendezvous endpoint not provided:
#   - let c10d select a free localhost port for single-node
#   - require explicit -e for multi-node
if [ -z "$rdzv_endpoint" ]; then
    if [ "$nnodes" -eq 1 ]; then
        rdzv_endpoint="localhost:0"
    else
        die "-e <rdzv_endpoint> is required when nnodes > 1."
    fi
fi

if [ ! -f "$config_file" ]; then
    die "config file '$config_file' does not exist."
fi

if [ ! -f "$train_script" ]; then
    die "training script '$train_script' does not exist."
fi

count_visible_gpus() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi

    local smi_output
    if ! smi_output=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); then
        echo 0
        return
    fi

    awk 'NF && $1 ~ /^[0-9]+$/ { count++ } END { print count + 0 }' <<<"$smi_output"
}

# --------------------------------------------------------------------------- #
# Auto-derive OMP_NUM_THREADS if not explicitly set
#   Heuristic:
#       total_cores = nproc (logical)
#       gpu_count   = number of visible GPUs (via nvidia-smi)
#       used_procs  = min(nproc_per_node, gpu_count) if gpu_count>0 else nproc_per_node
#       OMP = floor(total_cores / used_procs), at least 1
# --------------------------------------------------------------------------- #

if [ -z "$OMP_NUM_THREADS_VAL" ]; then
    if command -v nproc >/dev/null 2>&1; then
        total_cores=$(nproc)
    else
        total_cores=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || echo 32)
    fi
    gpu_count=$(count_visible_gpus)
    used_procs="$nproc_per_node"
    if [ "$gpu_count" -gt 0 ] && [ "$gpu_count" -lt "$used_procs" ]; then
        used_procs="$gpu_count"
    fi

    threads=$(( total_cores / used_procs ))
    if [ "$threads" -lt 1 ]; then
        threads=1
    fi

    OMP_NUM_THREADS_VAL="$threads"
fi

if ! [[ "$OMP_NUM_THREADS_VAL" =~ ^[1-9][0-9]*$ ]]; then
    die "-t <OMP_NUM_THREADS> must be a positive integer (got '$OMP_NUM_THREADS_VAL')."
fi

# --------------------------------------------------------------------------- #
# Environment setup (minimal, generic)
# --------------------------------------------------------------------------- #

export OMP_NUM_THREADS="$OMP_NUM_THREADS_VAL"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

# Ensure NumExpr thread settings never exceed NUMEXPR_MAX_THREADS (commonly 64).
numexpr_max="${NUMEXPR_MAX_THREADS:-64}"
if ! [[ "$numexpr_max" =~ ^[0-9]+$ ]]; then
    numexpr_max=64
fi

if [[ "${NUMEXPR_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
    numexpr_threads="$NUMEXPR_NUM_THREADS"
else
    numexpr_threads="$OMP_NUM_THREADS_VAL"
fi

if [ "$numexpr_threads" -gt "$numexpr_max" ]; then
    numexpr_threads="$numexpr_max"
fi

export NUMEXPR_MAX_THREADS="$numexpr_max"
export NUMEXPR_NUM_THREADS="$numexpr_threads"

echo "=== Distributed launch configuration ==="
echo "  launcher         : torchrun"
echo "  nnodes           : $nnodes"
echo "  nproc_per_node   : $nproc_per_node"
echo "  rdzv_endpoint    : $rdzv_endpoint"
echo "  rdzv_backend     : $rdzv_backend"
echo "  rdzv_id          : ${rdzv_id:-<none>}"
if [ "$rdzv_backend" = "static" ]; then
    echo "  node_rank        : $node_rank"
fi
echo "  train_script     : $train_script"
echo "  config_file      : $config_file"
echo "  total_cores      : ${total_cores:-unknown}"
echo "  gpu_count        : ${gpu_count:-unknown}"
echo "  OMP_NUM_THREADS  : $OMP_NUM_THREADS"
echo "  TORCH_NCCL_ASYNC_ERROR_HANDLING : ${TORCH_NCCL_ASYNC_ERROR_HANDLING:-unset}"
echo "  NCCL_P2P_DISABLE : ${NCCL_P2P_DISABLE:-unset}"
echo "  train_args       : ${train_args[*]:-<none>}"
echo "========================================"
echo

# --------------------------------------------------------------------------- #
# Build command
# --------------------------------------------------------------------------- #

cmd=(
    torchrun
    --nnodes "$nnodes"
    --nproc_per_node "$nproc_per_node"
)

if [ "$rdzv_backend" = "static" ]; then
    cmd+=(--node_rank "$node_rank")
elif [ -n "$rdzv_id" ]; then
    cmd+=(--rdzv_id "$rdzv_id")
fi

cmd+=(
    --rdzv_backend "$rdzv_backend"
    --rdzv_endpoint "$rdzv_endpoint"
    "$train_script"
    "${train_args[@]}"
)

echo "Command:"
printf '  %q ' "${cmd[@]}"
printf '\n\n'

if [ "$dry_run" -eq 1 ]; then
    echo "Dry run requested; exiting without execution."
    exit 0
fi

exec "${cmd[@]}"
