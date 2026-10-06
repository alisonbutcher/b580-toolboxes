#!/usr/bin/env bash
# qwen — dedicated launcher for the daily-driver setup: Qwen3.8-27B-Q4_K_S,
# VULKAN backend (measured faster than SYCL for this quant: 26.4 vs ~23-24
# t/s with MTP, Aug 31 2026), MTP speculative decoding, on 2x B580.
#
# This bypasses b580.sh entirely (no picker, no sidecar-file lookup) — the
# validated flags below are hardcoded so this script is self-contained and
# won't silently drift if the sidecar file at
# models/active/qwen3.8-27b-Q4_K_S.gguf.args ever
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
#                ceiling, but note that was validated on SYCL; Vulkan's
#                memory layout differs, so re-verify under real load before
#                trusting it here — the boundary hangs rather than cleanly
#                OOMs, see README)

set -euo pipefail

MODEL="${QWEN_MODEL:-$HOME/models/active/qwen3.8-27b-Q4_K_S.gguf}"
MMPROJ="${QWEN_MMPROJ:-$HOME/models/active/qwen3.8-27b-mmproj-F16.gguf}"
CTX="${QWEN_CTX:-110688}"
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
    --spec-type draft-mtp --spec-draft-n-max 2
    --alias "unsloth/Qwen3.8-27B-Q4_K_S"
)

[[ -n "$MMPROJ" && -e "$MMPROJ" ]] && ARGS+=(--mmproj "$MMPROJ")

echo "-> $TOOLBOX | $(basename "$MODEL") | ctx=$CTX"
exec toolbox run -c "$TOOLBOX" -- llama-server "${ARGS[@]}" "$@"
