# B580 llama.cpp Toolboxes (SYCL + Vulkan)

Reproducible Toolbx/Distrobox images for running llama.cpp on 2x Intel Arc
B580 (Battlemage) GPUs on Fedora. Built from these Containerfiles rather than
installed to the host, so driver/toolchain versions stay pinned and the host
stays clean.

## Build

```bash
podman build -t localhost/b580-llamacpp-sycl:latest   -f Containerfile.sycl .
podman build -t localhost/b580-llamacpp-vulkan:latest -f Containerfile.vulkan .
```

## Create + enter

```bash
toolbox create b580-sycl --image localhost/b580-llamacpp-sycl:latest \
  -- --device /dev/dri --group-add video --group-add render \
     --security-opt seccomp=unconfined
```

```bash
toolbox enter b580-sycl
```

```bash
toolbox create b580-vulkan --image localhost/b580-llamacpp-vulkan:latest \
  -- --device /dev/dri --group-add video --group-add render \
     --security-opt seccomp=unconfined
```

```bash
toolbox enter b580-vulkan
```

`llama-server`, `llama-bench`, `llama-cli` etc. are on `PATH` inside each
toolbox once entered.

## Running models with `b580.sh`

```bash
./b580.sh                              # interactive picker (requires fzf)
./b580.sh --list                       # list discovered models
./b580.sh -m PATH -b sycl|vulkan [-- server|bench] [extra llama-* args]
```

With no `-m`, it clears the screen and opens an `fzf` picker over every
`.gguf` under `$B580_MODEL_DIR` (default `~/models/gguf`). Each entry is
tagged `[dense]`/`[moe]` from the same filename heuristic used for backend
suggestion, and a preview pane at the bottom shows the full path of the
highlighted model. It then runs the picked model in the matching toolbox
(`llama-server` by default, or `llama-bench` via `-- bench`).

Backend suggestion (`-b`) follows the pattern below: dense models default to
SYCL, MoE models to Vulkan — override any time with `-b`. Drop a
`<model>.gguf.args` file next to a model to auto-load default flags for it
(e.g. MTP settings); see the comment header in `b580.sh` for details.

## Known Fedora 44 gotchas (already handled in the Containerfiles)

- `intel-level-zero` (GPU-side driver) and `oneapi-level-zero` (the loader
  `sycl-ls` actually calls) are **separate packages** — installing only the
  former leaves `sycl-ls` seeing zero L0 devices even though `clinfo` sees
  the GPU fine via OpenCL.
- Vulkan build needs `spirv-headers-devel`, `spirv-tools-devel`, and
  `glslang-devel` — cmake's Vulkan `find_package` fails on "SPIRV-Headers"
  without them, but the error doesn't mention which package to install.
- `intel-media-driver` isn't packaged for Fedora — not needed for compute,
  safe to skip.
- Fedora 44 ships dnf5, which dropped `config-manager --add-repo <url>` in
  favor of `config-manager addrepo`. Intel also doesn't host a ready-made
  `oneAPI.repo` file at a stable URL — write it directly (see
  `Containerfile.sycl`) rather than trying to fetch one.
- The SYCL backend needs oneMKL (`intel-oneapi-mkl-devel`) at build time —
  cmake fails with a `FindMKL.cmake` error without it. The full
  `intel-oneapi-base-toolkit` bundle isn't required; the compiler package
  (`intel-oneapi-compiler-dpcpp-cpp`) plus MKL devel is enough and keeps
  the image much smaller.

## Image size

| Image | Size |
|---|---|
| SYCL | 6.95 GB |
| Vulkan | 1.93 GB |

SYCL's image is ~3.6x larger — the oneAPI compiler toolchain and oneMKL
libraries dominate that, versus Vulkan's comparatively lightweight Mesa
driver + SPIR-V toolchain. Combined with the MoE benchmark results below,
this makes Vulkan the more efficient default for MoE-heavy workloads —
smaller image, faster to build, and faster at inference for that model
type.

## Benchmark findings (2x B580, `-ngl 999`, auto-split, single-stream)

| Model | Type | SYCL pp512 | SYCL tg128 | Vulkan pp512 | Vulkan tg128 | Winner |
|---|---|---|---|---|---|---|
| Gemma-4-12B (Q4_0) | Dense | 1287.9 | 41.9 | 569.4 | 35.2 | **SYCL**, ~2.3x pp |
| Qwen3.6-35B-A3B (Q3_K_S) | MoE | 716.5 | 40.7 | 1019.7 | 61.3 | **Vulkan**, ~42% pp / ~51% tg |
| gpt-oss-20B (Q4_K_S) | MoE | 836.5 | 34.6 | 1472.1 | 66.6 | **Vulkan**, ~76% pp / ~92% tg |
| Ornith-1.0-35B (Q3_K_M) | MoE | 788.6 | 43.0 | 991.0 | 56.8 | **Vulkan**, ~26% pp / ~32% tg |

**Pattern: dense models favor SYCL, MoE models favor Vulkan — consistently,
across three different MoE families.** Likely cause: SYCL's advantage comes
from matrix-core-optimized dense GEMM kernels; Vulkan's `KHR_coopmat` path
seems to handle MoE's sparse expert-routing/gather-scatter access pattern
better on current Battlemage drivers.

Practical takeaway: pick backend per model architecture, not as a single
global default.

## Known issue: manual `-ts` (tensor-split) ratios can hang on SYCL

A forced `-ts 40,60` split (uneven manual ratio) triggered a `UR_RESULT_
ERROR_OUT_OF_RESOURCES` / GPU engine timeout + reset on the `xe` kernel
driver (`ccs` compute engine) during a large MoE full-offload run. Automatic
split (`-ngl 999` with no `-ts` flag, or `-ts 1,1` even split) did not
reproduce this. Recommendation: let llama.cpp auto-split rather than forcing
custom ratios until this is root-caused upstream.

## MTP (Multi-Token Prediction) speculative decoding

Qwen3.6 ships MTP-capable GGUFs (`unsloth/Qwen3.6-27B-MTP-GGUF`) with the
extra draft heads baked into the single weights file — no separate draft
model needed. Enable with:

```
--spec-type draft-mtp --spec-draft-n-max 2
```

`-np 1` is required — parallel slots (`-np > 1`) aren't supported with MTP
yet.

### Auto-fit is unreliable with MTP — tune context manually

llama.cpp's automatic VRAM fitting (triggered when `-ngl` is left unset)
does **not** account for MTP's extra draft-context memory overhead. On
Qwen3.6-27B-Q4_K_S (SYCL, q8_0 KV cache, 2x B580), auto-fit picked
`n_ctx_slot = 44544` — smaller than the manually-tuned working value below —
and it still OOM'd under real inference load (crashed inside the MTP draft
decode step, not at load time). Manual binary-search on `-c` found a real,
load-tested working ceiling:

| `-c` | Result |
|---|---|
| 32768 | ✓ loads (4 slots by default — add `-np 1` for one long slot) |
| 65536 | ✓ loads, ✓ real inference tested clean |
| 81920 | ✓ loads, ✓ real inference tested clean — **recommended ceiling** |
| 98304 | loads, but OOMs during MTP draft decode under load |
| 131072 | fails at load |

Tested with: `-m Qwen3.6-27B-Q4_K_S.gguf -b sycl -c 81920 --cache-type-k
q8_0 --cache-type-v q8_0 -np 1 --spec-type draft-mtp --spec-draft-n-max 2`.

**Takeaway:** for MTP runs, don't rely on auto-fit (don't just omit `-ngl`
and hope) — set `-c` manually and validate with a real prompt, not just a
clean model load, since the failure mode only shows up under inference.
Auto-fit may still be fine for non-MTP runs; that wasn't separately tested.
