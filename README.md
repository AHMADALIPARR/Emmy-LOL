# LOL — Little Old Lady

A rope-memory PTX weaver for Emmy's Kernel IR, built for the
RiftStack Emmy Hackathon 2026.

The Apollo Guidance Computer's software lived in core rope memory —
programs woven by hand, thread by thread, through ferrite cores by
the women of Raytheon the engineers called the Little Old Ladies.
Read-only, dense, and final: once woven, the program *was* the
hardware. LOL weaves the same way. Every kernel here has a static
shape, so everything knowable before launch — tile geometry,
strides, unrolled bodies, constants — is woven into the PTX at
compile time, as immediates and straight-line code. Nothing at
runtime computes what the loom already knew.

## How it works

`compile.py` is a direct Kernel IR → PTX lowering. It loads the IR
with Emmy's own classes (`Graph.from_dict`) and walks the statement
tree, emitting one PTX translation per statement kind. Dispatch is
on the statement's type — never on the problem name or filename:
the weaver lowers what the tree says, it does not guess kernel
semantics.

- **Tiles** decomposed over grid and block, row-major, with the
  bounds guard derived from the tile's extents
- **Pointwise** loads/stores (grouped values woven as consecutive
  addresses), elementwise ops under their numpy/ufunc names
- **Reductions**: strided loops, in-place loop-carried accumulators
  (including the base-folded online-softmax step), warp-shuffle
  butterflies over coupled states, shared-memory partials, tree
  folds
- **GEMM**: multi-axis tiles, `cp.async` ring-buffer staging woven
  exactly as scheduled, dynamic shared memory when the buffers
  exceed the 48KB static limit, atomic epilogues
- **Fused + attention**: online softmax carried through a fused
  QK^T / softmax / PV kernel in registers

Anything the weaver cannot lower raises immediately — no silent
fallback, no half-woven kernel.

One detail cost a day and is worth writing down: Emmy seeds a
`maximum` accumulator with **-1e30**, a finite sentinel, not -inf.
Weave -inf instead and the softmax rescaling walks an inf − inf
path the reference never takes; the attention kernels land ~0.75
off. With the sentinel, they land at 6.6e-07.

## Results

All fifteen open problems pass on the hosted runner (RTX 5090),
checked with the task's own `submit.py`:

| category | problems | result |
|---|---|---|
| pointwise | p00–p03 | pass (exact) |
| reductions | r01–r04 | pass |
| matmul | m01–m02 | pass |
| fused | f01–f03 | pass |
| attention-like | a01–a02 | pass |

The hidden set is the same categories at unseen shapes, so the
weaver was also rehearsed against nine self-generated variants —
odd lengths, a row-mean, a third matmul shape, wider softmax rows,
tanh and sigmoid pointwise — minted with the organizers' own
generator (`rehearsal/`). All nine pass. That rehearsal caught two
generalizations the open set never forces: tile axes are assigned
from one global thread id (the renderer's own rule), and an
Accum's "base" operand folds the old accumulator into the step.

## Run it

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

Requires Python 3.12+ and `emmy-ml` (to load the IR). No GPU
needed — the hackathon's hosted runner assembles and executes.
Reproduce the full open set with `./check.sh` (expects the task
repository at `~/workspace/emmy-hackathon-2026` or `$EMMY_TASK_DIR`).

## Layout

- `compile.py` — the weaver
- `check.sh` — all fifteen open problems through the hosted runner
- `rehearsal/` — self-generated hidden-set variants and generator
