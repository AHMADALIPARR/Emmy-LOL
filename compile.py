#!/usr/bin/env python3
"""LOL -- Little Old Lady: a rope-memory PTX weaver for Emmy Kernel IR.

The Apollo Guidance Computer's software lived in core rope memory:
programs woven by hand, thread by thread, through ferrite cores --
read-only, dense, and final. LOL weaves the same way. Every Emmy
problem has a static shape, so everything knowable before launch --
tile geometry, strides, unrolled bodies, constants -- is woven into
the PTX at compile time, as immediates and straight-line code.
Nothing at runtime computes what the loom already knew.

Contract (fixed by the Emmy Hackathon task):

    python compile.py path/to/kernel.ir.json > out.ptx

The translation is a direct Kernel IR -> PTX lowering written for
this project. Emmy's own classes are used only to *load* the IR
(Graph.from_dict), never to render it.

Statement kinds are added category by category (pointwise first).
Anything the weaver does not know how to weave raises immediately --
no silent fallback, no half-lowered kernel.
"""

import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph

PTX_VERSION = "8.0"   # sm_89 needs >= 7.8
PTX_TARGET = "sm_89"


class WeaveError(Exception):
    """Raised for any IR the weaver cannot lower. Fail closed."""


class Weaver:
    def __init__(self):
        self.lines = []            # PTX body lines, in order
        self.reg_max = {"f": 0, "r": 0, "rd": 0, "p": 0}
        self.values = {}           # SSA name -> (reg, kind) kind in {f32, i32}
        self.buffers = {}          # buffer name -> %rd global address reg
        self.axis_vars = {}        # axis name -> %r reg (mutable: loop vars)
        self.smem = {}             # smem name -> (size_bytes, align, elem_bytes)
        self.smem_decls = []       # module-scope .shared declarations
        self.label_n = 0
        self.block = ()            # launch block dims from meta.json
        self.grid = ()             # launch grid dims from meta.json
        self.shapes = {}           # buffer name -> shape tuple (row-major)

    # -- register / label plumbing -------------------------------------

    def reg(self, kind):
        self.reg_max[kind] += 1
        return "%" + kind + str(self.reg_max[kind])

    def label(self, hint):
        self.label_n += 1
        return "L_%s_%d" % (hint, self.label_n)

    def emit(self, text):
        self.lines.append("        " + text)

    def emit_label(self, name):
        self.lines.append(name + ":")

    # -- expression compiler (integer index trees) ----------------------

    def expr_i32(self, e):
        """Compile an index expression tree to an %r register."""
        t = type(e).__name__
        if t == "Var":
            if e.name in self.axis_vars:
                return self.axis_vars[e.name]
            if e.name in self.values and self.values[e.name][1] == "i32":
                return self.values[e.name][0]
            raise WeaveError("unknown integer Var %r" % (e.name,))
        if t == "Literal":
            r = self.reg("r")
            self.emit("mov.u32 %s, %d;" % (r, int(e.value)))
            return r
        if t == "BinaryExpr":
            a = self.expr_i32(e.left)
            b = self.expr_i32(e.right)
            r = self.reg("r")
            ops = {"+": "add.u32", "-": "sub.u32", "*": "mul.lo.u32",
                   "/": "div.u32", "%": "rem.u32"}
            if e.op not in ops:
                raise WeaveError("index BinaryExpr op %r not woven" % (e.op,))
            self.emit("%s %s, %s, %s;" % (ops[e.op], r, a, b))
            return r
        raise WeaveError("index expression kind %s not woven" % t)

    def pred_of(self, e):
        """Compile a comparison expression to a %p predicate register."""
        t = type(e).__name__
        if t != "BinaryExpr" or e.op not in ("==", "!=", "<", "<=", ">", ">="):
            raise WeaveError("predicate kind %s/%s not woven"
                             % (t, getattr(e, "op", None)))
        a = self.expr_i32(e.left)
        b = self.expr_i32(e.right)
        p = self.reg("p")
        cmp_op = {"==": "eq", "!=": "ne", "<": "lt",
                  "<=": "le", ">": "gt", ">=": "ge"}[e.op]
        self.emit("setp.%s.s32 %s, %s, %s;" % (cmp_op, p, a, b))
        return p

    def flat_index(self, buffer, index_exprs, extra, shape):
        """Row-major flattened element index (%r) for an index tuple."""
        if len(index_exprs) == 1:
            idx = self.expr_i32(index_exprs[0])
        else:
            if shape is None or len(shape) != len(index_exprs):
                raise WeaveError("no shape for %d-D index into %r"
                                 % (len(index_exprs), buffer))
            idx = None
            for i, e in enumerate(index_exprs):
                stride = 1
                for d in shape[i + 1:]:
                    stride *= d
                r = self.expr_i32(e)
                if stride != 1:
                    m = self.reg("r")
                    self.emit("mul.lo.u32 %s, %s, %d;" % (m, r, stride))
                    r = m
                if idx is None:
                    idx = r
                else:
                    s = self.reg("r")
                    self.emit("add.u32 %s, %s, %s;" % (s, idx, r))
                    idx = s
        if extra:
            r = self.reg("r")
            self.emit("add.u32 %s, %s, %d;" % (r, idx, extra))
            idx = r
        return idx

    def address_of(self, buffer, index_exprs, extra):
        """Global address (%rd) of buffer[index + extra], f32 elements."""
        idx = self.flat_index(buffer, index_exprs, extra,
                              self.shapes.get(buffer))
        wide = self.reg("rd")
        self.emit("mul.wide.u32 %s, %s, 4;" % (wide, idx))
        addr = self.reg("rd")
        self.emit("add.u64 %s, %s, %s;" % (addr, self.buffers[buffer], wide))
        return addr

    # -- statement dispatch ---------------------------------------------

    def walk(self, stmts):
        for s in stmts:
            kind = type(s).__name__
            handler = getattr(self, "stmt_" + kind, None)
            if handler is None:
                raise WeaveError("statement kind %s not woven yet" % kind)
            handler(s)

    # -- statements ------------------------------------------------------

    def guard_axis(self, coord, extent, end):
        p = self.reg("p")
        self.emit("setp.ge.u32 %s, %s, %d;" % (p, coord, extent))
        self.emit("@%s bra %s;" % (p, end))

    def stmt_Tile(self, s):
        # Axis assignment, as the open problems demonstrate it:
        # a single axis takes the global thread coordinate along x
        # (blockIdx.x * blockDim.x + threadIdx.x), so multi-block
        # launches subdivide that one axis. With two axes, the outer
        # axis is the block index (blockIdx.x, one block per cell of
        # it) and the inner "_co" axis is the thread coordinate
        # within the block (r04: 64 rows on grid.x, 128 cooperating
        # threads per row).
        end = self.label("tile_end")
        if len(s.axes) == 1:
            ax = s.axes[0]
            extent = int(ax.extent.expr.value)
            tid = self.reg("r")
            self.emit("mov.u32 %s, %%tid.x;" % tid)
            coord = tid
            if self.block and self.block[0] > 1:
                cta = self.reg("r")
                self.emit("mov.u32 %s, %%ctaid.x;" % cta)
                base = self.reg("r")
                self.emit("mul.lo.u32 %s, %s, %d;"
                          % (base, cta, self.block[0]))
                coord = self.reg("r")
                self.emit("add.u32 %s, %s, %s;" % (coord, base, tid))
            self.axis_vars[ax.name] = coord
            self.guard_axis(coord, extent, end)
        elif len(s.axes) == 2:
            outer, inner = s.axes
            cta = self.reg("r")
            self.emit("mov.u32 %s, %%ctaid.x;" % cta)
            self.axis_vars[outer.name] = cta
            self.guard_axis(cta, int(outer.extent.expr.value), end)
            tid = self.reg("r")
            self.emit("mov.u32 %s, %%tid.x;" % tid)
            self.axis_vars[inner.name] = tid
            self.guard_axis(tid, int(inner.extent.expr.value), end)
        else:
            raise WeaveError("Tile with %d axes not woven yet"
                             % len(s.axes))
        self.walk(list(s.body))
        self.emit_label(end)

    def stmt_Load(self, s):
        if s.dtype.name != "f32":
            raise WeaveError("Load dtype %s not woven" % s.dtype.name)
        for i, name in enumerate(s.names):
            r = self.reg("f")
            if s.input in self.smem:
                addr = self.smem_address_of(s.input, s.index, i)
                self.emit("ld.shared.f32 %s, [%s];" % (r, addr))
            elif s.input in self.buffers:
                addr = self.address_of(s.input, s.index, i)
                self.emit("ld.global.f32 %s, [%s];" % (r, addr))
            else:
                raise WeaveError("Load from unknown buffer %r" % (s.input,))
            self.values[name] = (r, "f32")

    def smem_address_of(self, buffer, index_exprs, extra):
        """Shared-memory address (%r, 32-bit) of buffer[index + extra]."""
        shape = self.smem[buffer][3] if len(self.smem[buffer]) > 3 else None
        idx = self.flat_index(buffer, index_exprs, extra, shape)
        addr = self.reg("r")
        self.emit("mov.u32 %s, %s;" % (addr, buffer))
        off = self.reg("r")
        self.emit("add.u32 %s, %s, %s;" % (off, idx, idx))   # 2*idx
        self.emit("add.u32 %s, %s, %s;" % (off, off, off))   # 4*idx
        self.emit("add.u32 %s, %s, %s;" % (addr, addr, off))
        return addr

    def stmt_Assign(self, s):
        if s.dtype is not None and s.dtype.name != "f32":
            raise WeaveError("Assign dtype %s not woven" % s.dtype.name)
        op = s.op.name
        regs = []
        for a in s.args:
            if a not in self.values:
                raise WeaveError("Assign arg %r not an SSA value" % (a,))
            regs.append(self.values[a][0])
        out = self.reg("f")
        # Emmy names elementwise ops after the numpy ufuncs (with a
        # few torch spellings mixed in); both spellings are woven.
        binary = {"add": "add.rn.f32", "sub": "sub.rn.f32",
                  "subtract": "sub.rn.f32",
                  "mul": "mul.rn.f32", "multiply": "mul.rn.f32",
                  "div": "div.rn.f32", "divide": "div.rn.f32",
                  "maximum": "max.f32", "minimum": "min.f32"}
        unary = {"neg": "neg.f32", "negative": "neg.f32",
                 "abs": "abs.f32", "absolute": "abs.f32",
                 "sqrt": "sqrt.rn.f32", "rsqrt": "rsqrt.approx.f32"}
        if op in binary and len(regs) == 2:
            self.emit("%s %s, %s, %s;" % (binary[op], out, regs[0], regs[1]))
        elif op in unary and len(regs) == 1:
            self.emit("%s %s, %s;" % (unary[op], out, regs[0]))
        elif op == "relu" and len(regs) == 1:
            self.emit("mov.f32 %s, 0f00000000;" % out)
            self.emit("max.f32 %s, %s, %s;" % (out, regs[0], out))
        elif op == "exp" and len(regs) == 1:
            # exp(x) = ex2(x * log2(e)); 0f3FB8AA3B is log2(e) in f32 hex.
            self.emit("mul.rn.f32 %s, %s, 0f3FB8AA3B;" % (out, regs[0]))
            self.emit("ex2.approx.f32 %s, %s;" % (out, out))
        elif op == "recip" and len(regs) == 1:
            self.emit("mov.f32 %s, 0f3F800000;" % out)  # 1.0
            self.emit("div.rn.f32 %s, %s, %s;" % (out, out, regs[0]))
        else:
            raise WeaveError("elementwise op %r (arity %d) not woven"
                             % (op, len(regs)))
        self.values[s.name] = (out, "f32")

    def stmt_Write(self, s):
        if s.value_dtype.name != "f32":
            raise WeaveError("Write dtype %s not woven" % s.value_dtype.name)
        for i, vname in enumerate(s.values):
            if vname not in self.values:
                raise WeaveError("Write value %r not an SSA value" % (vname,))
            reg = self.values[vname][0]
            if s.output in self.smem:
                if s.atomic:
                    raise WeaveError("atomic shared Write not woven")
                if not s.index:
                    # Scalar shared cell: index () means element 0.
                    addr = self.reg("r")
                    self.emit("mov.u32 %s, %s;" % (addr, s.output))
                else:
                    addr = self.smem_address_of(s.output, s.index, i)
                self.emit("st.shared.f32 [%s], %s;" % (addr, reg))
            elif s.output in self.buffers:
                if not s.index:
                    addr = self.buffers[s.output]
                else:
                    addr = self.address_of(s.output, s.index, i)
                if s.atomic:
                    self.emit("red.global.add.f32 [%s], %s;" % (addr, reg))
                else:
                    self.emit("st.global.f32 [%s], %s;" % (addr, reg))
            else:
                raise WeaveError("Write to unknown buffer %r" % (s.output,))

    # -- reduction machinery (category 2) --------------------------------

    ACCUM_IDENTITY = {"add": "0f00000000", "multiply": "0f3F800000",
                      "maximum": "0fFF800000", "minimum": "0f7F800000"}

    def init_accum(self, s):
        if s.name in self.values:
            return
        op = s.op.name
        if op not in self.ACCUM_IDENTITY:
            raise WeaveError("Accum op %r has no woven identity" % (op,))
        r = self.reg("f")
        self.emit("mov.f32 %s, %s;" % (r, self.ACCUM_IDENTITY[op]))
        self.values[s.name] = (r, "f32")

    def stmt_Accum(self, s):
        self.init_accum(s)
        if s.value not in self.values:
            raise WeaveError("Accum value %r not an SSA value" % (s.value,))
        acc_reg = self.values[s.name][0]
        val_reg = self.values[s.value][0]
        op = s.op.name
        ptx = {"add": "add.rn.f32", "multiply": "mul.rn.f32",
               "maximum": "max.f32", "minimum": "min.f32"}.get(op)
        if ptx is None:
            raise WeaveError("Accum op %r not woven" % (op,))
        # Loop-carried state: the accumulator must be updated IN
        # PLACE, in its permanent register. Rebinding to a fresh
        # register (the straight-line Assign move) would emit one
        # instruction that re-reads the identity every iteration.
        self.emit("%s %s, %s, %s;" % (ptx, acc_reg, acc_reg, val_reg))

    def stmt_StridedLoop(self, s):
        if s.unroll:
            raise WeaveError("unrolled StridedLoop not woven yet")
        body = list(s.body)
        for st in body:
            if type(st).__name__ == "Accum":
                self.init_accum(st)
        var = self.reg("r")
        start = self.expr_i32(s.start)
        self.emit("mov.u32 %s, %s;" % (var, start))
        axis_name = s.axis.name
        prev = self.axis_vars.get(axis_name)
        self.axis_vars[axis_name] = var
        head = self.label("loop")
        end = self.label("loop_end")
        self.emit_label(head)
        p = self.reg("p")
        if s.end is not None:
            bound = self.expr_i32(s.end)
            self.emit("setp.ge.s32 %s, %s, %s;" % (p, var, bound))
        else:
            extent = int(s.axis.extent.expr.value)
            self.emit("setp.ge.s32 %s, %s, %d;" % (p, var, extent))
        self.emit("@%s bra %s;" % (p, end))
        self.walk(body)
        step = int(s.step.value)
        self.emit("add.u32 %s, %s, %d;" % (var, var, step))
        self.emit("bra %s;" % head)
        self.emit_label(end)
        if prev is None:
            del self.axis_vars[axis_name]
        else:
            self.axis_vars[axis_name] = prev

    def stmt_WarpShuffle(self, s):
        if len(s.state) != 1 or len(s.state_b) != 1:
            raise WeaveError("multi-state WarpShuffle not woven yet")
        state, state_b = s.state[0], s.state_b[0]
        if state not in self.values:
            raise WeaveError("WarpShuffle state %r undefined" % (state,))
        delta = s.length // 2
        while delta >= 1:
            cur = self.values[state][0]
            bits = self.reg("r")
            self.emit("mov.b32 %s, %s;" % (bits, cur))
            shuf = self.reg("r")
            self.emit("shfl.sync.bfly.b32 %s, %s, %d, 0x%x, 0xffffffff;"
                      % (shuf, bits, delta, s.length - 1))
            other = self.reg("f")
            self.emit("mov.b32 %s, %s;" % (other, shuf))
            self.values[state_b] = (other, "f32")
            self.walk(list(s.combine_states))
            delta //= 2

    def stmt_Smem(self, s):
        n = 1
        for e in s.extents:
            n *= int(e.expr.value) if hasattr(e, "expr") else int(e)
        size = n * 4
        align = int(s.align) or 4
        shape = tuple(int(e.expr.value) if hasattr(e, "expr") else int(e)
                      for e in s.extents)
        self.smem[s.name] = (size, align, 4, shape)
        decl = ".shared .align %d .b8 %s[%d];" % (align, s.name, size)
        if decl not in self.smem_decls:
            self.smem_decls.append(decl)

    def stmt_Sync(self, s):
        if s.warp:
            self.emit("bar.warp.sync 0xffffffff;")
        elif s.count:
            self.emit("bar.sync %d, %d;" % (int(s.barrier_id), int(s.count)))
        else:
            self.emit("bar.sync %d;" % int(s.barrier_id))

    def stmt_Cond(self, s):
        p = self.pred_of(s.cond)
        body = list(s.body)
        other = list(s.else_body) if s.else_body else []
        if not other:
            end = self.label("cond_end")
            self.emit("@!%s bra %s;" % (p, end))
            self.walk(body)
            self.emit_label(end)
        else:
            then_l = self.label("cond_then")
            end = self.label("cond_end")
            self.emit("@%s bra %s;" % (p, then_l))
            self.walk(other)
            self.emit("bra %s;" % end)
            self.emit_label(then_l)
            self.walk(body)
            self.emit_label(end)

    def stmt_TreeHalve(self, s):
        if len(s.bufs) != 1 or len(s.state) != 1 or len(s.state_b) != 1:
            raise WeaveError("multi-buffer TreeHalve not woven yet")
        buf = s.bufs[0]
        state, state_b = s.state[0], s.state_b[0]
        # Fold the length shared-memory partials into state. Every
        # thread performs the same read-only fold; the guarded Write
        # that follows decides who stores the result. (The reference
        # PTX guards the fold to thread 0 instead -- equivalent, since
        # the fold has no side effects.)
        for i in range(int(s.length)):
            r = self.reg("f")
            idx = self.reg("r")
            self.emit("mov.u32 %s, %d;" % (idx, i))
            addr = self.reg("r")
            self.emit("mov.u32 %s, %s;" % (addr, buf))
            off = self.reg("r")
            self.emit("add.u32 %s, %s, %s;" % (off, idx, idx))
            self.emit("add.u32 %s, %s, %s;" % (off, off, off))
            self.emit("add.u32 %s, %s, %s;" % (addr, addr, off))
            self.emit("ld.shared.f32 %s, [%s];" % (r, addr))
            if i == 0:
                self.values[state] = (r, "f32")
            else:
                self.values[state_b] = (r, "f32")
                self.walk(list(s.combine_states))


def find_kernel(graph):
    nodes = [n for n in graph.nodes.values()
             if type(n.op).__name__ == "KernelOp"]
    if len(nodes) != 1:
        raise WeaveError("expected exactly 1 KernelOp, found %d" % len(nodes))
    return nodes[0]


def meta_for(ir_path, node):
    """meta.json shipped beside the IR is the launch contract:
    parameter order and block geometry both come from it.
    Fall back to graph inputs + outputs when it is absent."""
    meta_path = Path(ir_path).parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for k in meta.get("kernels", []):
            if k.get("entry") == node.op.name:
                shapes = {a["name"]: tuple(a.get("shape", ()))
                          for a in k.get("args", [])}
                return (list(k["arg_order"]), tuple(k.get("block", ())),
                        tuple(k.get("grid", ())), shapes)
    return (list(node.inputs) + [t.name for t in node.outputs],
            (), (), {})


def uses_builtins(stmts, found):
    for s in stmts:
        for attr in ("body", "else_body"):
            sub = getattr(s, attr, None)
            if sub:
                uses_builtins(list(sub), found)
        for attr in ("index", "cond", "start"):
            e = getattr(s, attr, None)
            exprs = e if isinstance(e, tuple) else (e,)
            for x in exprs:
                scan_vars(x, found)


def scan_vars(e, found):
    if e is None:
        return
    if type(e).__name__ == "Var" and e.name in ("lane", "warp"):
        found.add(e.name)
    for attr in ("left", "right"):
        scan_vars(getattr(e, attr, None), found)


def compile_kernel(ir_path):
    graph = Graph.from_dict(json.loads(Path(ir_path).read_text()))
    node = find_kernel(graph)
    kernel = node.op
    order, block, grid, shapes = meta_for(ir_path, node)

    w = Weaver()
    w.block = block
    w.grid = grid
    w.shapes = dict(shapes)
    for n in graph.nodes.values():
        for t in getattr(n, "outputs", ()) or ():
            shape = tuple(int(d.expr.value) if hasattr(d, "expr")
                          else int(d) for d in t.shape)
            w.shapes.setdefault(t.name, shape)
    body_stmts = list(kernel.body)
    builtins = set()
    uses_builtins(body_stmts, builtins)

    # Prologue: buffers arrive as .u64 params; convert once to
    # generic global addresses and keep them in w.buffers.
    for name in order:
        raw = w.reg("rd")
        w.emit("ld.param.u64 %s, [param_%s];" % (raw, name))
        glob = w.reg("rd")
        w.emit("cvta.to.global.u64 %s, %s;" % (glob, raw))
        w.buffers[name] = glob
    if builtins:
        tid = w.reg("r")
        w.emit("mov.u32 %s, %%tid.x;" % tid)
        if "lane" in builtins:
            lane = w.reg("r")
            w.emit("and.b32 %s, %s, 31;" % (lane, tid))
            w.values["lane"] = (lane, "i32")
        if "warp" in builtins:
            warp = w.reg("r")
            w.emit("shr.u32 %s, %s, 5;" % (warp, tid))
            w.values["warp"] = (warp, "i32")
    w.walk(body_stmts)
    w.emit("ret;")

    out = []
    out.append("// LOL -- Little Old Lady rope-memory weaver.")
    out.append("// Kernel IR -> PTX, woven at compile time. Entry and")
    out.append("// parameter order follow the problem's meta.json contract.")
    out.append(".version %s" % PTX_VERSION)
    out.append(".target %s" % PTX_TARGET)
    out.append(".address_size 64")
    out.append("")
    for d in w.smem_decls:
        out.append(d)
    params = ",\n".join("        .param .u64 param_%s" % n for n in order)
    out.append(".visible .entry %s(" % kernel.name)
    out.append(params)
    out.append(")")
    out.append("{")
    if w.reg_max["p"]:
        out.append("        .reg .pred %%p<%d>;" % (w.reg_max["p"] + 1))
    if w.reg_max["f"]:
        out.append("        .reg .f32 %%f<%d>;" % (w.reg_max["f"] + 1))
    if w.reg_max["r"]:
        out.append("        .reg .b32 %%r<%d>;" % (w.reg_max["r"] + 1))
    if w.reg_max["rd"]:
        out.append("        .reg .b64 %%rd<%d>;" % (w.reg_max["rd"] + 1))
    out.append("")
    out.extend(w.lines)
    out.append("}")
    return "\n".join(out) + "\n"


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: python compile.py <kernel.ir.json>\n")
        return 2
    try:
        ptx = compile_kernel(argv[1])
    except WeaveError as e:
        sys.stderr.write("LOL weave error: %s\n" % e)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
