#!/usr/bin/env bash
# b580 — pick a GGUF and a backend, run it inside the matching toolbox.
#
# Usage:
#   b580                              interactive picker (requires fzf)
#   b580 --list                       list discovered models
#   b580 -m PATH -b sycl|vulkan [-- server|bench] [extra llama-* args]
#
# Interactive picker: fzf list of models, each tagged [dense]/[moe] from the
# same filename heuristic used for backend suggestion; the preview pane at
# the bottom shows the full path of the highlighted model.
#
# Backend auto-suggestion is based on the benchmark findings in README.md:
# dense models generally do better on SYCL, MoE models on Vulkan. This is a
# suggestion only — override with -b.
#
# Per-model default args: drop a file named "<model>.gguf.args" next to any
# model (plain text, one or more whitespace-separated llama-server/-bench
# flags, comments starting with # ignored) and it's loaded automatically
# whenever that model is picked — interactively or via -m. Anything passed
# after -- on the command line is appended after the sidecar args, so CLI
# flags can override sidecar ones.
#
# Example sidecar for an MTP model:
#   Qwen3.6-27B-Q4_K_S.gguf.args
#   -c 81920 --cache-type-k q8_0 --cache-type-v q8_0 -np 1
#   --spec-type draft-mtp --spec-draft-n-max 2

set -euo pipefail

MODEL_DIR="${B580_MODEL_DIR:-$HOME/models/gguf}"
MODE="server"        # server | bench
BACKEND=""
MODEL_PATH=""
CLI_EXTRA_ARGS=()

usage() {
    grep '^#' "$0" | cut -c3-
    exit 1
}

list_models() {
    # Exclude mmproj/vision-projector files — these are auxiliary vision
    # encoders, not standalone models, and llama-server will error with
    # "unsupported model architecture: 'clip'" if picked as -m by mistake.
    find "$MODEL_DIR" -type f -name '*.gguf' ! -iname '*mmproj*' 2>/dev/null | sort
}

# crude dense-vs-MoE guess from the filename/path — good enough as a nudge,
# not a substitute for knowing your own model.
detect_model_type() {
    local path_lower
    path_lower="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    if [[ "$path_lower" == *"a3b"* || "$path_lower" == *"moe"* || "$path_lower" == *"gpt-oss"* ]]; then
        echo "moe"
    else
        echo "dense"
    fi
}

suggest_backend() {
    if [[ "$(detect_model_type "$1")" == "moe" ]]; then
        echo "vulkan"
    else
        echo "sycl"
    fi
}

# Read "<model>.gguf.args" next to the model, if present. Strips # comments,
# splits on whitespace. Empty/missing file -> no extra args.
sidecar_args() {
    local args_file="$1.args"
    local -a out=()
    if [[ -f "$args_file" ]]; then
        # shellcheck disable=SC2046
        out=($(grep -v '^\s*#' "$args_file" | tr '\n' ' '))
    fi
    printf '%s\n' "${out[@]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list) list_models; exit 0 ;;
        -m) MODEL_PATH="$2"; shift 2 ;;
        -b) BACKEND="$2"; shift 2 ;;
        --) shift; MODE="${1:-server}"; shift || true; CLI_EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "Unknown arg: $1"; usage ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    mapfile -t MODELS < <(list_models)
    if [[ ${#MODELS[@]} -eq 0 ]]; then
        echo "No .gguf files found under $MODEL_DIR (set B580_MODEL_DIR to override)."
        exit 1
    fi
    command -v fzf >/dev/null 2>&1 || { echo "fzf is required for the interactive picker (or pass -m directly)."; exit 1; }

    clear

    # Each fzf line is "TAG label [sidecar]\tfullpath" — --with-nth hides the
    # tab-separated full path from the list but --preview can still read it.
    # Label is the repo dir (relative to MODEL_DIR) rather than the filename,
    # e.g. "unsloth/Qwen3.6-27B-MTP-GGUF" — the repo dir often carries info
    # the filename doesn't (MTP variants, etc), and it's what collapses
    # split GGUF shards (Model-00001-of-00003.gguf, -00002-, -00003-) into a
    # single entry instead of listing every part. Root-level flat files (no
    # subdir under MODEL_DIR) fall back to the filename since there's no repo
    # dir to use. Multiple distinct non-split files sharing a repo dir (e.g.
    # separate quants side by side) keep one entry each, disambiguated with
    # the filename appended.
    # ANSI colors need --ansi on fzf's side to render instead of showing raw
    # escape codes; each colored segment resets before the tab so nothing
    # bleeds into the hidden full-path field.
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; RESET=$'\033[0m'
    SHARD_RE='-[0-9]{5}-of-[0-9]{5}\.gguf$'

    declare -A SHARD_CANON=()      # dirname -> first (alphabetically) shard part
    declare -A NONSHARD_COUNT=()   # dirname -> count of non-shard files in it
    for path in "${MODELS[@]}"; do
        dir="$(dirname "$path")"
        if [[ "$(basename "$path")" =~ $SHARD_RE ]]; then
            [[ -z "${SHARD_CANON[$dir]:-}" ]] && SHARD_CANON[$dir]="$path"
        elif [[ "$dir" != "$MODEL_DIR" ]]; then
            NONSHARD_COUNT[$dir]=$(( ${NONSHARD_COUNT[$dir]:-0} + 1 ))
        fi
    done

    declare -A SHARD_DIR_EMITTED=()
    FZF_LINES=()
    for path in "${MODELS[@]}"; do
        dir="$(dirname "$path")"
        fname="$(basename "$path")"
        if [[ "$fname" =~ $SHARD_RE ]]; then
            [[ "$path" != "${SHARD_CANON[$dir]}" ]] && continue
            [[ -n "${SHARD_DIR_EMITTED[$dir]:-}" ]] && continue
            SHARD_DIR_EMITTED[$dir]=1
            label="$(dirname "${path#"$MODEL_DIR"/}")"
        elif [[ "$dir" == "$MODEL_DIR" ]]; then
            label="$fname"
        else
            label="$(dirname "${path#"$MODEL_DIR"/}")"
            (( ${NONSHARD_COUNT[$dir]:-0} > 1 )) && label="${label} · ${fname}"
        fi

        marker=""
        [[ -f "${path}.args" ]] && marker="${DIM} [has sidecar args]${RESET}"
        if [[ "$(detect_model_type "$path")" == "moe" ]]; then
            tag="${YELLOW}[moe]${RESET}"
        else
            tag="${GREEN}[dense]${RESET}"
        fi
        FZF_LINES+=("${tag} ${label}${marker}"$'\t'"$path")
    done

    # --height=100% forces full-terminal use even if the user's shell config
    # sets a shorter default via FZF_DEFAULT_OPTS (a later flag wins).
    SELECTED="$(printf '%s\n' "${FZF_LINES[@]}" | fzf \
        --ansi --height=100% \
        --delimiter=$'\t' --with-nth=1 \
        --prompt="model> " \
        --header="Pick a model under $MODEL_DIR (tag is a filename heuristic, not a guarantee)" \
        --color=header:italic:dim,prompt:cyan,pointer:cyan,marker:cyan \
        --preview='echo {2}' \
        --preview-window=down,3,border-top,wrap)"

    [[ -z "$SELECTED" ]] && { echo "No model selected."; exit 1; }
    MODEL_PATH="$(cut -f2 <<<"$SELECTED")"
fi

if [[ -z "$BACKEND" ]]; then
    SUGGESTED="$(suggest_backend "$MODEL_PATH")"
    SUGGESTED_LETTER="${SUGGESTED:0:1}"
    read -rp "Backend [s]ycl/[v]ulkan (suggested: $SUGGESTED_LETTER): " BACKEND
    BACKEND="${BACKEND:-$SUGGESTED_LETTER}"
fi

case "$BACKEND" in
    s|sycl)   BACKEND="sycl";   TOOLBOX="b580-sycl" ;;
    v|vulkan) BACKEND="vulkan"; TOOLBOX="b580-vulkan" ;;
    *) echo "Backend must be 's'/'sycl' or 'v'/'vulkan'"; exit 1 ;;
esac

mapfile -t SIDECAR_ARGS < <(sidecar_args "$MODEL_PATH")
ALL_EXTRA_ARGS=("${SIDECAR_ARGS[@]}" "${CLI_EXTRA_ARGS[@]}")

echo "-> $TOOLBOX | $(basename "$MODEL_PATH") | mode=$MODE"
[[ ${#SIDECAR_ARGS[@]} -gt 0 ]] && echo "   sidecar args: ${SIDECAR_ARGS[*]}"

case "$MODE" in
    server)
        toolbox run -c "$TOOLBOX" -- llama-server -m "$MODEL_PATH" -ngl 999 "${ALL_EXTRA_ARGS[@]}"
        ;;
    bench)
        toolbox run -c "$TOOLBOX" -- llama-bench -m "$MODEL_PATH" -ngl 999 "${ALL_EXTRA_ARGS[@]}"
        ;;
    *)
        echo "Mode must be 'server' or 'bench'"; exit 1
        ;;
esac
