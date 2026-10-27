# LOL — Little Old Lady

A rope-memory PTX weaver for Emmy Kernel IR — RiftStack Emmy Hackathon 2026.

> Every kernel here has a static shape. So everything knowable before
> launch — tile geometry, strides, constants — is woven into the PTX
> at compile time, as immediates and straight-line code. Nothing at
> runtime computes what the loom already knew.

**▶ Watch the walkthrough (3:32, in the creator's own voice):
https://vimeo.com/1239809455/10afd44756**

## The idea

Apollo's software lived in core rope memory — programs woven by hand,
thread by thread, through ferrite cores by the women of Raytheon the
engineers called the Little Old Ladies. Read-only, dense, final: once
woven, the program *was* the hardware. LOL weaves the same way.

## What it does

`compile.py` lowers Emmy Kernel IR straight to PTX. It loads the IR
with Emmy's own classes (`Graph.from_dict`) and walks the statement
tree, emitting one PTX translation per statement kind. Dispatch is on
the statement's type — never on the problem name or filename. The
weaver lowers what the tree says; it never guesses kernel semantics.
Anything it cannot lower raises immediately — no silent fallback, no
half-woven kernel.

- **Pointwise**: grouped values woven as consecutive addresses,
  elementwise ops under their numpy/ufunc names
- **Reductions**: strided loops, in-place loop-carried accumulators
  (including the base-folded online-softmax step), warp-shuffle
  butterflies over coupled states, shared-memory partials, tree folds
- **GEMM**: multi-axis tiles, `cp.async` ring-buffer staging woven
  exactly as scheduled, dynamic shared memory past the 48KB static
  limit, atomic epilogues
- **Fused + attention**: online softmax carried through a fused
  QK^T / softmax / PV kernel in registers

## Results — 15/15 open, 9/9 rehearsal

All fifteen open problems pass on the hosted runner (RTX 5090),
checked with the task's own `submit.py`. Re-verified today: 15/15,
0 failed.

| category | problems | result |
|---|---|---|
| pointwise | p00–p03 | pass (exact) |
| reductions | r01–r04 | pass |
| matmul | m01–m02 | pass |
| fused | f01–f03 | pass |
| attention-like | a01–a02 | pass |

Example: p00 pointwise add — max abs err 0.0, ~32 µs, 22 registers.

The hidden set is the same categories at unseen shapes, so the
weaver was also rehearsed against nine self-generated variants —
odd lengths, a row-mean, a third matmul shape, wider softmax rows,
tanh and sigmoid pointwise — minted with the organizers' own
generator (`rehearsal/`). All nine pass. That rehearsal caught two
generalizations the open set never forces: tile axes are assigned
from one global thread id (the renderer's own rule), and an Accum's
"base" operand folds the old accumulator into the step.

## One detail worth knowing: -1e30

Emmy seeds a `maximum` accumulator with **-1e30**, a finite sentinel,
not -inf. Weave -inf instead and the softmax rescaling walks an
inf − inf path the reference never takes; the attention kernels land
~0.75 off. With the sentinel, they land at 6.6e-07. That is the kind
of thing you only find by running the reference, not by reading the
spec.

## Run it

```bash
python compile.py path/to/kernel.ir.json > out.ptx
./check.sh   # full open set through the hosted runner
```

Requires Python 3.12+ and `emmy-ml` (to load the IR). No GPU needed
locally — the hosted runner assembles and executes. `./check.sh`
expects the task repository at `~/workspace/emmy-hackathon-2026` or
`$EMMY_TASK_DIR`.

## Layout

- `compile.py` — the weaver
- `check.sh` — all fifteen open problems through the hosted runner
- `rehearsal/` — self-generated hidden-set variants and generator
