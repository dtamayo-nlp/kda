# KDA training step (forward + backward) with a per-channel decay gate on Hopper

## Request

We train models with Kimi Delta Attention (KDA) layers on NVIDIA GH200 (Hopper, `sm_90`). Every KDA layer runs FLA's Triton `chunk_kda` forward and its autograd backward, which is the best implementation we have on Hopper.

We ask for a faster training step on Hopper: forward, backward, and the gradients of the decay-gate parameters. The kernel has to run on GH200; the wishlist lists B200 and B300, and this request targets Hopper as agreed with the KDA team. The shapes are 32 and 28 heads, and 32 heads at 8192 tokens matters most. The 16384 and 32768 workloads cover longer sequences.

## Contract

`run(q, k, v, g, beta, A_log, dt_bias, cu_seqlens, do) -> (o, dq, dk, dv, dg, dbeta, dA_log, ddt_bias)`

| Tensor | Shape | Dtype | Meaning |
|---|---|---|---|
| `q`, `k` | `[1, T, H, 128]` | bf16 | Queries and keys, L2-normalized inside the kernel (eps 1e-6) |
| `v` | `[1, T, H, 128]` | bf16 | Values |
| `g` | `[1, T, H, 128]` | bf16 | Raw decay pre-activation |
| `beta` | `[1, T, H]` | bf16 | Beta logits, sigmoid inside the kernel |
| `A_log` | `[H*128]` | fp32 | **Per-channel** decay scale: one value per head and key channel |
| `dt_bias` | `[H*128]` | fp32 | Per-channel decay bias |
| `cu_seqlens` | `[N+1]` | int64 | Packed document boundaries |
| `do` | `[1, T, H, 128]` | bf16 | Upstream gradient of `o` |

Semantics, per document, with the state reset to zero at every boundary in `cu_seqlens`:

```
q_t = l2norm(q_t) / sqrt(128);   k_t = l2norm(k_t);   b_t = sigmoid(beta_t)
a_t = -5 * sigmoid(exp(A_log) * (g_t + dt_bias))        # log decay, per channel, in [-5, 0)
S'_t = diag(exp(a_t)) S_{t-1}
S_t = S'_t + b_t k_t (v_t - S'_t^T k_t)^T
o_t = S_t^T q_t
```

The outputs are `o` and the gradients of every input given `do`. They are bf16, except `dA_log` and `ddt_bias`, which are fp32 sums over all tokens.

The contract above has no initial or final state, and the workloads do not benchmark that path. A kernel that can also take an initial state for the first document and return the final state of the last document, with their gradients, as FLA's `chunk_kda` does with `initial_state` and `output_final_state=True`, is welcome; every other document boundary still resets the state to zero.

Differences from the 2026-09-27 KDA kernels:

1. `A_log` is per channel, `[H*128]`, instead of per head, `[H]`. This is the only change to the forward math.
2. The backward pass is required, including `dA_log` and `ddt_bias`. The published TIRx backward stops at the gradient of the activated gate, and FLA's gate step turns that into `dA_log` and `ddt_bias`. The FLA fork below already does that step per channel, so a kernel that keeps the same split is fine.
3. The `q`/`k` and `v` head counts are equal, and the head dimension is 128 everywhere.

A single forward-and-backward entry point that recomputes its own intermediates is welcome. flashinfer-bench calls `run` under `torch.no_grad()`, so a solution that uses autograd has to enable it inside `run`, as `baseline.py` does.

## Workloads

All workloads use the packed (THD) layout. The two `_packed` workloads pack regular documents. The `_long` ones cut documents of at least 2048 tokens, so their first document starts before the window and their last one continues after it. `h32_t8192_gaterange` exercises the gate at its operating points: a third of the channels have `dt_bias = 8`, so their log decay sits at the -5 bound for almost every token, a third have `dt_bias = 0`, so it spreads over the whole range, a quarter have `dt_bias = -12`, and the rest are at initialization; its documents include one-token and sub-64-token ones. `h32_t32768_longmem` is one 32768-token document that tests long-range state (see below).

| Workload | H | T | Documents |
|---|---:|---:|---:|
| `h32_t8192_packed` | 32 | 8192 | 10 |
| `h28_t8192_packed` | 28 | 8192 | 15 |
| `h32_t16384_long` | 32 | 16384 | 6 |
| `h28_t16384_long` | 28 | 16384 | 7 |
| `h32_t32768_long` | 32 | 32768 | 11 |
| `h28_t32768_long` | 28 | 32768 | 12 |
| `h32_t8192_gaterange` | 32 | 8192 | 17 |
| `h32_t32768_longmem` | 32 | 32768 | 1 |

`q`, `k`, `v`, `g`, `beta` and `do` are random normal, except `beta` in `h32_t32768_longmem`. `cu_seqlens`, `A_log` and `dt_bias` come from one file per workload in `data/`, and so does that `beta`. FlashInfer Trace inputs can only be `random`, `scalar` or `safetensors`, and random integers are not valid document boundaries, so the boundaries are stored. Document lengths follow a realistic distribution. A quarter of the channels have `dt_bias = -12`, which gives almost no decay. With random keys and `sigmoid(beta)` near 0.5, the delta rule still overwrites the state within a few hundred tokens, so on the other workloads a kernel that drops state older than 2048-4096 tokens passes both checks. `h32_t32768_longmem` therefore stores `beta` logits drawn from N(-4, 1), so `sigmoid(beta)` is near 0.03 and the state keeps information for more than 8192 tokens. There, a kernel that keeps only the last 8192-16384 tokens of state is 5% off, and one that restarts the state every 16384 tokens is 24% off.

## Correctness

The reference in `definition.json` is an fp32 token-by-token recurrence differentiated with PyTorch autograd. It is slow and uses gradient checkpointing to stay within memory. Measured errors of the FLA baseline against it:

| Workload | Worst relative RMSE, any output | Relative RMSE, `dA_log` and `ddt_bias` | Worst fraction outside atol = rtol = 5e-2 | flashinfer-bench |
|---|---:|---:|---:|---|
| `h32_t8192_packed` | 0.59% | 0.56% | 0.46% | PASSED |
| `h28_t8192_packed` | 0.58% | 0.55% | 0.42% | PASSED |
| `h32_t16384_long` | 0.60% | 0.59% | 1.17% | PASSED |
| `h28_t16384_long` | 0.60% | 0.59% | 1.17% | PASSED |
| `h32_t32768_long` | 0.59% | 0.56% | 1.76% | PASSED |
| `h28_t32768_long` | 0.62% | 0.62% | 1.56% | PASSED |
| `h32_t8192_gaterange` | 1.09% | 1.09% | 0.27% | PASSED |
| `h32_t32768_longmem` | 0.64% | 0.48% | 0.10% | PASSED |

All outputs are finite. Every output except `dA_log` and `ddt_bias` stays within atol = rtol = 1e-2, except for one element of `dg` in each 28-head long workload, with absolute error about 0.012. `dA_log` and `ddt_bias` are fp32 sums over every token, so single elements with heavy cancellation show large relative error even when the whole vector is accurate. The flashinfer-bench element gate in `benchmark.py` is therefore atol = rtol = 5e-2 with `required_matched_ratio = 0.95`. That gate alone is loose: `o`, `dq`, `dk` and `dv` have a standard deviation near 0.03 and `dg` near 0.003, so a forward output that is 50% off everywhere, or outputs rounded to fp8, still pass it while the gradients are right. `benchmark.py` therefore adds a strict check on a fresh random draw of every workload: the relative RMSE of every bf16 output must stay at or below 1% and that of `dA_log` and `ddt_bias` at or below 2% (the FLA baseline is at 0.5-0.7% on the bf16 outputs and up to 1.2% on `ddt_bias`), and every element must lie within max(0.5 x RMS of the reference output, 5% of the reference element, 1/32 of the largest reference magnitude), with no allowance for outliers. The last term is there for `dg`, whose RMS is 0.003 while its largest elements exceed 1. Random keys are nearly orthogonal, so the intra-chunk triangular solve barely changes the result on them: replacing `(I + L)^-1` with `I - L + L^2` passes both checks on every workload. The strict check therefore runs a second time on the three 8192-token workloads with each key repeated 8 times in a row. That shortcut is then more than 50% off, while FLA stays at or below 0.82% on the bf16 outputs and 1.21% on `ddt_bias`. A candidate passes only if both checks pass.

## Baseline

`baseline.py` imports `chunk_kda` from [swiss-ai/flash-linear-attention @ 1820dba7](https://github.com/swiss-ai/flash-linear-attention/commit/1820dba7e15fb927294779f23ac5097eb3927b25), installed as shown under Environment. That fork is upstream FLA v0.5.2 plus one commit that adds the per-channel `A_log` to the decay-gate kernels in `fla/ops/kda/gate.py`. Upstream FLA reads `A_log` per head and would silently index the wrong values with a per-channel tensor, so `baseline.py` checks that the fork is installed. FLA is MIT licensed.

Measured on one NVIDIA GH200 120GB GPU (Hopper, `sm_90`, 132 SMs) in the environment below. The FLA columns are the median of 20 timed calls after 5 warm-up calls.

| Workload | FLA forward (ms) | FLA forward + backward (ms) | flashinfer-bench latency, forward + backward (ms) |
|---|---:|---:|---:|
| `h32_t8192_packed` | 1.28 | 4.98 | 5.38 |
| `h28_t8192_packed` | 1.18 | 4.52 | 4.90 |
| `h32_t16384_long` | 2.13 | 9.33 | 9.64 |
| `h28_t16384_long` | 1.93 | 8.35 | 8.73 |
| `h32_t32768_long` | 3.92 | 17.91 | 18.19 |
| `h28_t32768_long` | 3.49 | 16.10 | 16.38 |
| `h32_t8192_gaterange` | 1.37 | 5.17 | 5.58 |
| `h32_t32768_longmem` | 4.20 | 18.50 | 18.78 |

The backward pass is about three quarters of the time.

## Environment

The NGC PyTorch image `nvcr.io/nvidia/pytorch:25.12-py3` already has PyTorch and Triton. Add the FLA fork and the benchmark tools:

```bash
docker run --gpus all -it -v $PWD:/work -w /work nvcr.io/nvidia/pytorch:25.12-py3
python -m pip install einops "transformers>=4.45"
python -m pip install --no-deps "git+https://github.com/swiss-ai/flash-linear-attention@1820dba7e15fb927294779f23ac5097eb3927b25"
python -m pip install flashinfer-bench==0.1.2 safetensors
```

We validated this on one GH200 in this exact image: Python 3.12.3, PyTorch 2.10.0a0+b4e4ee81d3.nv25.12, Triton 3.5.1, CUDA 13.1. We ran the image with a different container runtime, so the install lines and the evaluation were tested, but not the `docker run` line itself.

## Evaluate

`benchmark.py` registers `baseline.py::run` as a Python solution and runs flashinfer-bench on every workload. It resolves the `data/` paths relative to the request directory. It skips timing the reference, which is a slow token-by-token loop, so compare candidates against the baseline latency rather than the reported `speedup_factor`. It uses 5 warm-up runs, 20 timed iterations and 1 trial, then runs the strict check described under Correctness on a fresh random draw and on the repeated-key draw (each seed is printed; set `STRICT_SEED` to repeat a draw). It prints one `strict ... PASS` line per workload and output.

```bash
CUDA_VISIBLE_DEVICES=0 REQUEST_DIR=requests/dtamayo-nlp-kda-training \
    python requests/dtamayo-nlp-kda-training/benchmark.py
```

To evaluate a candidate, change the solution's `entry_point` and `sources` in `benchmark.py`, and its `language` and `destination_passing_style` if it is not a Python function that returns its outputs. The run prints `ALL_OK` when every workload passes both checks. The reference takes about 20 seconds per 8192 tokens, and the whole run about 12 minutes on one GH200.

## License

The definition, workloads and wrapper are Apache-2.0 code, and this documentation is Creative Commons Attribution 4.0, under the [project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline calls the separately installed swiss-ai fork of flash-linear-attention, which is MIT licensed. The `data/` files hold synthetic gate parameters, beta logits and document boundaries only.
