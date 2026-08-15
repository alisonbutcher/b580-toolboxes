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

With no `-m`, it clears the screen and opens a full-height, colored `fzf`
picker over every `.gguf` under `$B580_MODEL_DIR` (default `~/models/gguf`).
Each entry is tagged `[dense]`/`[moe]` (green/yellow) from the same filename
heuristic used for backend suggestion, and a preview pane at the bottom
shows the full path of the highlighted model. It then runs the picked model
in the matching toolbox (`llama-server` by default, or `llama-bench` via
`-- bench`).

Entries are labeled by repo directory, not filename — e.g.
`unsloth/Qwen3.6-27B-MTP-GGUF` rather than `Qwen3.6-27B-Q4_K_S.gguf` — since
the repo dir often carries info the filename doesn't (MTP variants, etc).
Root-level flat files with no repo dir fall back to showing the filename.
Split GGUF shards (`Model-00001-of-00003.gguf`, `-00002-`, `-00003-`) in the
same repo dir collapse into a single entry, resolving to the first part;
distinct quants sitting side by side in the same repo dir instead keep one
entry each, with the filename appended to tell them apart.

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

## Qwen3.8-27B-Q4_K_S findings (2x B580, SYCL)

Same dense/MTP-capable family as Qwen3.6-27B-MTP above, but noticeably more
VRAM-efficient — every number here beats the 3.6 findings at the same
context. Baseline settings throughout: `--cache-type-k q8_0 --cache-type-v
q8_0 -np 1 -fa on`, `-ngl 999` (auto-split, no `-ts`).

### Non-MTP baseline

With `-c` omitted (`--fit on`, no `-ngl` override needed since `b580.sh`
already passes `-ngl 999` and lets `common_fit_params` size context):

| Quant | `n_ctx_slot` (auto-fit) | tg | Notes |
|---|---|---|---|
| Q4_K_S | 161280 | 19.79 t/s | matches Unsloth Studio's own tuned Vulkan fork on speed (~18-20 t/s) but ~1.8x the context (88k there) |
| IQ4_XS | 171264 | 14.30 t/s | IQ-format costs real compute even on SYCL (~28% slower than Q4_K_S) — but nowhere near Vulkan's collapse to 5.54 t/s on the same quant. SYCL's IQ-quant kernels are meaningfully better optimized than Vulkan's. |

pp512-equivalent throughput on a real ~8500-token prompt: ~490-541 tok/s —
the tiny-prompt number you'll see on a first request (~10 t/s) is pure fixed
overhead noise, not a real pp measurement. Always validate pp with a prompt
of at least a couple thousand tokens.

### MTP binary search — and a new failure mode: hangs, not just crashes

| `-c` | Result |
|---|---|
| 65536 | ✓ loads, ✓ real inference tested clean (24.28 t/s, 62.5% draft acceptance) |
| 81920 | ✓ loads, ✓ real inference tested clean (23.34 t/s, 58.4% draft acceptance) — matches 3.6's recommended ceiling |
| 98304 | ✓ loads, ✓ **survived a heavy ~8700-token stress prompt** (22.89 t/s, 65.2% acceptance) — this is exactly where the 3.6 model OOM'd under load; 3.8 doesn't |
| 114688 | ✓ loads, ✓ survived the same stress prompt — best numbers of the whole run (24.43 t/s, 72.7% acceptance) — **recommended ceiling** |
| 118784 | ✗ **hangs** — `UR_RESULT_ERROR_OUT_OF_RESOURCES` / `Error OP CONCAT`, process pegs a CPU core and never responds. Not a clean abort like the others; needs `kill -9`. Confirmed no lasting GPU damage (`sycl-ls` still sees both devices immediately after), but don't rely on that — kill it promptly rather than waiting it out. |
| 122880 | ✗ clean OOM crash (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, inside `common_speculative_impl_draft_mtp::process`) |
| 131072 | ✗ clean OOM crash, same signature — 3.6 failed here at *load* time; 3.8 gets further (loads fine, only fails under heavy decode load) |

Tested with: `-m Qwen3.8-27B-Q4_K_S.gguf -b sycl -c 114688 --cache-type-k
q8_0 --cache-type-v q8_0 -np 1 -fa on --spec-type draft-mtp
--spec-draft-n-max 2`. Stress-validated with a real ~8700-token prompt, not
just a short completion — the light-load test alone wasn't enough to catch
the 118784 hang (a quick 150-token completion loaded and generated fine at
every context size tried, including ones that failed under heavier load).

**New takeaway beyond the 3.6 findings:** the boundary right above a working
MTP context isn't a clean cliff — failure mode is inconsistent (some values
OOM cleanly, one hung instead). Don't binary-search all the way to the exact
edge and stop one step below the last failure; leave real margin (3.8's
114688 sits comfortably below the 118784 hang, not immediately adjacent to
it) since the step immediately below a crash can still be the unstable one.

Sidecar in use: `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_S.gguf.args`.
