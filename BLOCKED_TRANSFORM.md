# Blocked 3D Transform — Corner-Turn Algorithm

A data-movement-conscious restructuring of the MRA 3D transform, designed so
each of K thread-block-equivalents ("wavefronts" on AMD) owns one K×K slab of
the tensor throughout the computation. Passes 1 and 2 are fully local per
block; pass 3 requires a single all-to-all "corner turn" that transposes the
block-level ownership from `k'` to `a`.

Reference implementations
- CPU (C++): `validate.hip` → `cpu_transform3d_blocked`
- NumPy: `mra_python/algorithms.py` → `transform_nd_blocked`

Both match `cpu_transform3d` / `transform_nd` bit-exactly (CPU) or to
floating-point noise (NumPy).

---

## The canonical operation

We compute the 3D transform

```
result[a, b, c] = Σ_{i, j, k} A[i, j, k] · B[i, a] · B[j, b] · B[k, c]
```

via three sequential contractions, one per tensor axis. Standard formulations
apply this as three K²×K GEMMs over the whole tensor, reloading the input
through HBM on every pass. The blocked algorithm keeps each K×K slab
resident in registers/LDS of a single wavefront for all three passes, paying
one inter-wave exchange instead of three HBM round-trips.

---

## Visual language

A consistent set of conventions used across every frame:

- **Axis colors.** `i'/a` red, `j'/b` green, `k'/c` blue.
- **Blocks are K×K squares.** Row-bar and column-bar outside each square
  name the axis currently playing that role.
- **Solid box = data, dashed box = operation.**
- **Arrows carry data.**

---

## Frame 1 — canonical operation

> ![Frame 1](frames/frame1_canonical.png)

A K×K×K input cube, a K×K matrix `B`, and a K×K×K output cube, with three
curved arrows labeled "contract `i'`", "contract `j'`", "contract `k'`".
Provides grounding; this is what all later frames compute.

---

## Frame 2 — distribute (slice along `k'`)

> ![Frame 2](frames/frame2_distribute.png)

The input cube is sliced along the fastest axis into K independent K×K
squares. Block `s` owns the slice `A[:, :, k'=s]`, with row = `i'` and
col = `j'`.

This establishes the row-of-K-squares motif used in every subsequent frame.

---

## Frame 3 — passes 1 and 2 (local GEMMs)

> ![Frame 3](frames/frame3_local_gemms.png)

Two stages stacked vertically on one frame:

```
row of K squares  ──(× B^T on left)──►  row of K squares       [Pass 1]
 rows: i',  cols: j'                     rows: a,   cols: j'

row of K squares  ──(× B on right)──►   row of K squares       [Pass 2]
 rows: a,   cols: j'                     rows: a,   cols: b
```

Both passes are **local** — no wavefront touches another wavefront's data.
A small "✓ local" badge on each block emphasizes this.

---

## Frame 4 — the corner turn (the money slide)

> ![Frame 4](frames/frame4_corner_turn.png)

The critical data-movement step. Shown as two rows of K blocks with colored
rows flowing between them.

**Before** (top): K blocks, each with rows colored red/green/blue/...
(= `a = 0, 1, 2, ...`). Every block has the same row-color pattern because
`a` is the within-block row index.

**After** (bottom): K blocks, each entirely one color. Block 0 all red,
block 1 all green, block 2 all blue. `a` is now the **block** index and
`k'` has become the within-block row.

The operation is a **block-level transpose of a K×K super-matrix** whose
cells are K-vectors indexed by `b`.

Callout on the frame: *"block-level transpose: block index ↔ within-block
row."* This is the single sentence the audience should leave with.

The main visual shows only the flow of one color (e.g. red rows, all
destined for block `a=0`) to keep the arrow count manageable; an inset
shows the full K=2 picture with all four arrows.

---

## Frame 5 — pass 3 + un-shuffle

> ![Frame 5](frames/frame5_pass3.png)

```
row of K squares  ──(× B^T on left)──►  row of K squares  ──(in-block ᵀ)──►  canonical result
 rows: k',  cols: b                      rows: c,   cols: b                   rows: b,  cols: c
```

Points to emphasize:

- Block index throughout this frame is `a` (inherited from the corner turn).
- The final transpose is **inside each K×K block only** — no inter-block
  traffic. On GPU this fuses into the pass-3 store.

---

## Frame 6 — the bookkeeping table (takeaway slide)

> ![Frame 6](frames/frame6_table.png)

| stage              | block idx | within-block row | within-block col |
|--------------------|-----------|------------------|------------------|
| distribute         | k'        | i'                | j'               |
| after pass 1       | k'        | **a**             | j'               |
| after pass 2       | k'        | a                 | **b**            |
| after corner turn  | **a**     | **k'**            | b                |
| after pass 3       | a         | **c**             | b                |
| after un-shuffle   | a         | **b**             | **c**            |

Bold cells mark which slot changed on each row. Two annotations:

- Rows 1–3 and 5: *"local GEMM: row slot updates."*
- Row 4: *"corner turn: block slot ↔ row slot."*

---

## Cost summary

| item                                 | cost per transform                    |
|--------------------------------------|---------------------------------------|
| compute (mathematical minimum)       | 3 · 2 · K⁴ FLOPs                      |
| HBM reads of A                       | K³ doubles (once, at distribute)      |
| HBM writes of result                 | K³ doubles (once, at un-shuffle)      |
| HBM reads of B                       | K² doubles (once, cached in LDS)      |
| inter-wave exchange (corner turn)    | K³ doubles (via LDS, one pass only)   |

Comparison against the naive 3-GEMM implementation is the whole point: HBM
traffic for A drops from 3·K³ to 1·K³, and the compute path is exactly the
same three K²×K contractions — just re-organized.

---

## K=2 walkthrough (self-checking)

Run `./validate -debug 1` in the build directory. Uses the counting tensor
`A[i, j, k] = 100i + 10j + k` and `B = I`, so every GEMM is a no-op and only
the corner turn produces visible motion. Each stage dumps all K blocks; the
final line is self-checking (`MATCH` / `DIFFER`).

Particularly useful as a "does this still work" smoke test when porting to
GPU.

---

## References

- `validate.hip` — CPU reference and GPU L1 correctness check
- `mra_python/algorithms.py::transform_nd_blocked` — NumPy version
- `transform.h`, `transform_level{2,3,4}.h` — GPU optimization levels 1–4
  (global memory → LDS → register blocking → MFMA); the blocked algorithm
  is the natural successor to L4
