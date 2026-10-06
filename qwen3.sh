#!/usr/bin/env bash
# qwen3 — dedicated launcher for Qwen3.6-35B-A3B-UD-IQ4_XS (MoE),
# VULKAN backend (faster than SYCL for MoE per local testing), on 2x B580.
#
# No MTP: this GGUF ships no nextn tensors, so no --spec-type flags here.
# Sampling defaults (temp etc.) come from the GGUF's embedded
# general.sampling.* metadata — no flags needed.
#
# Usage:
#   ./qwen3.sh                    # launch with the defaults below
#   ./qwen3.sh --port 8090        # override/add any llama-server flag
#
# Env overrides:
#   QWEN_MODEL   path to the UD-IQ4_XS gguf (default: below)
#   QWEN_MMPROJ  path to the vision mmproj (default: below — note it's the
#                F32 one, ~1.8GB VRAM; set empty to disable vision and spend
#                that on context instead)
#   QWEN_CTX     context size (default: 49152 — deliberately conservative:
#                17GB model + 1.8GB mmproj on 24GB total leaves limited KV
#                headroom, and this ceiling is NOT stress-tested. Tune upward
#                gradually per README method — near the limit these setups
#                hang rather than cleanly OOM)

set -euo pipefail

MODEL="${QWEN_MODEL:-$HOME/models/gguf/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
MMPROJ="${QWEN_MMPROJ:-$HOME/models/gguf/unsloth/Qwen3.6-35B-A3B-GGUF/mmproj-F32.gguf}"
CTX="${QWEN_CTX:-49152}"
TOOLBOX="b580-vulkan"

if [[ ! -e "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    echo "Set QWEN_MODEL, or symlink it there — see README.md." >&2
    exit 1
fi

if ! toolbox list 2>/dev/null | grep -q "$TOOLBOX"; then
    echo "Toolbox '$TOOLBOX' doesn't exist yet — see README.md 'Create + enter'." >&2
    exit 1
fi

ARGS=(
    -m "$MODEL"
    -ngl 999
    --split-mode layer
    --cache-type-k q8_0 --cache-type-v q8_0
    -np 1
    -fa on
    -c "$CTX"
    --context-shift
    --alias "unsloth/Qwen3.6-35B-A3B-UD-IQ4_XS"
)

[[ -n "$MMPROJ" && -e "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")

echo "-> $TOOLBOX | $(basename "$MODEL") | ctx=$CTX"
exec toolbox run -c "$TOOLBOX" -- llama-server "${ARGS[@]}" "$@"
