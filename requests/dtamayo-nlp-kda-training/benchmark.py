# SPDX-License-Identifier: Apache-2.0
"""Evaluate a solution entry point with flashinfer-bench (see README.md, Evaluate)."""
import os, sys
from pathlib import Path
import torch
from flashinfer_bench import Benchmark, BenchmarkConfig
from flashinfer_bench.bench.evaluators.utils import allocate_outputs, normalize_result
from flashinfer_bench.bench.utils import gen_inputs, load_safetensors
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, Trace, TraceSet

MAX_REL_RMSE = 0.01       # bf16 outputs (o, dq, dk, dv, dg, dbeta); the FLA baseline is at 0.5-0.8%
MAX_REL_RMSE_SUM = 0.02   # dA_log, ddt_bias: fp32 sums over all tokens with heavy cancellation
ELEM_RMS_FRAC = 0.5   # every element within max(0.5 * RMS(reference output), 5% of |reference|, ...)
ELEM_REL = 0.05
ELEM_ULP_FLOOR = 2.0 ** -5  # ... 1/32 of the largest |reference|, 4-8 bf16 ulps of it (dg is heavy-tailed: RMS 0.003, max ~1)
REPEAT_KEYS = 8  # second strict pass on the 8192-token workloads with each key repeated 8 times in a row


def strict_check(definition, workloads, solution, device="cuda:0", repeat_keys=1):
    """Relative-RMSE gate per output plus an RMS-scaled element gate with no allowance.

    The flashinfer-bench element gate (atol = rtol = 5e-2) is loose next to the scale of o, dq, dk, dv and dg,
    so a candidate must also pass this check on freshly drawn random inputs. Random keys are nearly orthogonal, so
    the intra-chunk triangular solve barely matters on them; with repeat_keys > 1 each key is repeated that many
    times in a row, which makes it matter.
    """
    registry = BuilderRegistry.get_instance()
    ref = registry.build_reference(definition)
    sol = registry.build(definition, solution)
    seed = int(os.environ.get("STRICT_SEED", torch.seed() % 2**31))
    print(f"strict seed {seed}", flush=True)
    ok = True
    for t in workloads:
        torch.manual_seed(seed)
        st = load_safetensors(definition, t.workload)
        inp = gen_inputs(definition, t.workload, device, safe_tensors=st)
        if repeat_keys > 1:
            k = inp[list(definition.inputs).index("k")]
            k.copy_(k[:, torch.arange(k.shape[1], device=k.device) // repeat_keys * repeat_keys])
        with torch.no_grad():
            ref_out = normalize_result(definition, ref(*inp), device)
            if sol.metadata.destination_passing_style:
                sol_out = allocate_outputs(definition, inp, device)
                sol(*inp, *sol_out)
            else:
                sol_out = normalize_result(definition, sol(*inp), device)
        torch.cuda.synchronize(device)
        for name, x, y in zip(definition.outputs, sol_out, ref_out):
            x, y = x.float(), y.float()
            rel_rmse = float((x - y).norm() / y.norm().clamp_min(1e-30))
            floor = torch.maximum(ELEM_RMS_FRAC * y.pow(2).mean().sqrt(), ELEM_ULP_FLOOR * y.abs().max())
            tol = torch.maximum(floor, ELEM_REL * y.abs())
            worst = float(((x - y).abs() / tol).max())
            max_rel = MAX_REL_RMSE_SUM if name in ("dA_log", "ddt_bias") else MAX_REL_RMSE
            passed = rel_rmse <= max_rel and worst <= 1.0 and bool(torch.isfinite(x).all())
            ok &= passed
            print(f"strict {t.workload.uuid} repeat_keys={repeat_keys} {name:9s} rel_rmse={rel_rmse:.4f} worst_elem={worst:.3f} "
                  f"max_abs_err={float((x - y).abs().max()):.4g} ref_absmax={float(y.abs().max()):.4g} "
                  f"{'PASS' if passed else 'FAIL'}", flush=True)
    return ok


def main():
    request = Path(os.environ["REQUEST_DIR"]).resolve()
    definition = Definition.model_validate_json((request / "definition.json").read_text())
    workloads = [Trace.model_validate_json(l) for l in (request / "workloads.jsonl").read_text().splitlines() if l.strip()]
    assert workloads and all(t.definition == definition.name for t in workloads)
    # safetensors paths are relative to the request directory
    for t in workloads:
        for spec in t.workload.inputs.values():
            if spec.type == "safetensors" and not os.path.isabs(spec.path):
                spec.path = str(request / spec.path)
    baseline = Solution(
        name=f"{definition.name}_baseline", definition=definition.name, author="baseline",
        spec=BuildSpec(language="python", target_hardware=["GH200"],
                       entry_point="baseline.py::run", destination_passing_style=False),
        sources=[SourceFile(path="baseline.py", content=(request / "baseline.py").read_text())])
    dataset = TraceSet(definitions={definition.name: definition}, workloads={definition.name: workloads},
                       solutions={definition.name: [baseline]})
    benchmark = Benchmark(dataset, BenchmarkConfig(
        warmup_runs=5, iterations=20, num_trials=1, rtol=5e-2, atol=5e-2, required_matched_ratio=0.95,
        profile_baseline=False, timeout_seconds=3600))
    try:
        results = benchmark.run_all(dump_traces=False)
    finally:
        benchmark.close()
    traces = results.traces.get(definition.name, [])
    for t in traces:
        print(t.model_dump_json())
    assert len(traces) == len(workloads), "Missing evaluation results"
    assert all(t.is_successful() for t in traces), "Correctness or execution failed"
    assert strict_check(definition, workloads, baseline), "Strict correctness check failed"
    short = [t for t in workloads if t.workload.axes["total_tokens"] == 8192]
    assert strict_check(definition, short, baseline, repeat_keys=REPEAT_KEYS), "Strict check with repeated keys failed"
    print("ALL_OK")


if __name__ == "__main__":
    main()
