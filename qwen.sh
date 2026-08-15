#!/usr/bin/env bash
# qwen — dedicated launcher for the daily-driver setup: Qwen3.8-27B-Q4_K_S,
# SYCL backend, MTP speculative decoding, on 2x B580.
#
# This bypasses b580.sh entirely (no picker, no sidecar-file lookup) — the
# validated flags below are hardcoded so this script is self-contained and
# won't silently drift if the sidecar file at
# models/gguf/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf.args ever
# changes. See README.md "Qwen3.8-27B-Q4_K_S findings" for how these numbers
# were derived (binary-searched context ceiling, real-load stress-tested,
# not just checked at model-load time).
#
# Usage:
#   ./qwen.sh                    # launch with the validated defaults
#   ./qwen.sh --port 8090        # override/add any llama-server flag
#
# Env overrides:
#   QWEN_MODEL   path to the Q4_K_S gguf (default: below)
#   QWEN_MMPROJ  path to the vision mmproj (default: below, set empty to
#                disable vision and free ~0.9GB + activation overhead)
#   QWEN_CTX     context size (default: 114688 — the stress-tested MTP
#                ceiling; see README before raising this, the boundary
#                right above it hangs rather than cleanly OOMs)

set -euo pipefail

MODEL="${QWEN_MODEL:-$HOME/models/gguf/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf}"
MMPROJ="${QWEN_MMPROJ:-$HOME/models/gguf/unsloth/Qwen3.8-27B-GGUF/mmproj-F16.gguf}"
CTX="${QWEN_CTX:-114688}"
TOOLBOX="b580-sycl"

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
    --cache-type-k q8_0 --cache-type-v q8_0
    -np 1
    -fa on
    -c "$CTX"
    --spec-type draft-mtp --spec-draft-n-max 2
)

[[ -n "$MMPROJ" && -e "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")

echo "-> $TOOLBOX | $(basename "$MODEL") | ctx=$CTX"
exec toolbox run -c "$TOOLBOX" -- llama-server "${ARGS[@]}" "$@"
