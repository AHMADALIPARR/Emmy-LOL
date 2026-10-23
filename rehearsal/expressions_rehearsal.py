"""Hidden-set rehearsal specs: same five categories as the open set,
different shapes and op mixes -- the variation the judges promise.
Format copied from the task repo's tools/expressions_open.py."""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


OPEN = [
    # pointwise: odd sizes (guard stress) and unseen ops
    dict(id="h01_pw_odd_add",
         code="torch.randn(1000) + torch.randn(1000)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="h02_pw_tanh_scale",
         code="torch.tanh(torch.randn(3000)) * 1.5 - 0.25",
         ref=lambda x: [(np.tanh(x.astype(np.float64)) * 1.5 - 0.25).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="h03_pw_sigmoid",
         code="torch.sigmoid(torch.randn(2049))",
         ref=lambda x: [(1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    # reductions: odd length, row means
    dict(id="h04_sum_odd",
         code="torch.randn(4097).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)],
         rtol=1e-3, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="h05_row_mean",
         code="torch.randn(48, 384).mean(-1)",
         ref=lambda x: [x.astype(np.float64).mean(-1).astype(np.float32)],
         rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    # matmul: a third shape/schedule
    dict(id="h06_matmul_rect2",
         code="torch.randn(64, 128) @ torch.randn(128, 96)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    # fused: wider softmax rows, rmsnorm at another width
    dict(id="h07_softmax_wide",
         code="torch.softmax(torch.randn(16, 1024), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-5),
    dict(id="h08_rmsnorm_512",
         code="x = torch.randn(128, 512); w = torch.randn(512); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=lambda x, w: [(x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5)
                            * w).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    # attention at a smaller size
    dict(id="h09_attention_small",
         code="q = torch.randn(32, 32); kt = torch.randn(32, 32); v = torch.randn(32, 32); "
              "torch.softmax(q @ kt / 5.656854, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 5.656854)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
]
