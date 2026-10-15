# SPDX-License-Identifier: Apache-2.0
"""Current training path: FLA chunk_kda (Triton) forward + autograd backward.

Requires the swiss-ai fork of flash-linear-attention (see README.md). Upstream FLA reads
A_log per head; with a per-channel A_log it would index the wrong values silently.
"""
import inspect

import torch
from fla.ops.kda import chunk_kda
from fla.ops.kda import gate as _fla_gate

assert "PER_CHANNEL" in inspect.getsource(_fla_gate), (
    "This FLA build has no per-channel A_log support; install swiss-ai/flash-linear-attention@1820dba7"
)


def run(q, k, v, g, beta, A_log, dt_bias, cu_seqlens, do):
    with torch.enable_grad():
        leaves = [x.detach().requires_grad_() for x in (q, k, v, g, beta, A_log, dt_bias)]
        q_, k_, v_, g_, b_, A_, d_ = leaves
        o, _ = chunk_kda(
            q_, k_, v_, g_, b_,
            A_log=A_, dt_bias=d_,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        dq, dk, dv, dg, dbeta, dA, ddt = torch.autograd.grad(o, leaves, grad_outputs=do)
    out = o, dq, dk, dv, dg, dbeta, dA.float(), ddt.float()
    return tuple(t.detach() for t in out)
