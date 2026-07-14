# Volta

> ⚠️ This is not the code that was used for the paper. That code was written by Kshitij Dubey at MSR and they have not agreed to released it. Hopefully, this reconstruction is useful; it appears to work on all the benchmarks from the paper. I've been quite busy, so some parts were written almost entirely by Fable 5 and I did not have time to do extensive quality assurance. ⚠️

Volta is a data race and equivalence checker for NVIDIA GPU kernels, implementing the approach from ["Equivalence Checking of ML GPU Kernels"](https://arxiv.org/pdf/2511.12638). Given a reference kernel implementation and an optimized counterpart, Volta proves their semantic equivalence over the reals, i.e., that they produce identical outputs for all valid inputs modulo floating point error, thereby verifying the correctness of the optimized kernel.

## Features

- **Deadlock Detection**: Identify deadlocks arising from over synchronization
- **Data Race Detection**: Identify races arising from under synchronization
- **Equivalence Checking**: Verify that optimized GPU kernels are semantically equivalent to their reference implementations
- **Two-kernel equivalence from the CLI**: `volta compare` checks a reference/optimized pair directly, without going through `volta_bench`
- **VC dump/replay**: persist the verification conditions from one run and rerun just the equivalence check later, skipping parse/lower/symbolic-execution entirely
- **Per-run logging and execution profiling**: every run gets a log file, and a per-instruction-kind execution profile is shown by default
- **Z3 backend**: check the same verification conditions with Z3 instead of the built-in decision procedure, for a "decides vs. cannot decide" timing/capability comparison

## How It Works

Volta has two phases:

1. **Symbolic Execution**: Executes both kernels symbolically (round-robin over all threads of CTA 0), tracking memory accesses and synchronization to detect data races and deadlocks and producing symbolic expressions representing output values as functions of input tensors.

2. **Equivalence Checking**: Verifies that the symbolic expressions from both kernels are mathematically equal over the reals. Each output element canonicalizes to a rational function whose polynomials are sums `c * monomial * e^{poly}` terms with exact rational coefficients. An optional `f64` oracle (`--verify-numeric`) re-checks every verdict at seeded random inputs.

## Soundness and Completeness

Equivalence checking treats floating-point values as reals. Within that model:

- Race and deadlock detection is sound and complete for structured-CTAs (see
  [requirements](#requirements)) using `+`, `-`, `*`, `/`, `exp`, and `max`/`min`
  with symmetric CTAs (only CTA 0 is checked, but note that the grid size still
  matters for index computations).

- `sqrt`, `log`, `abs`, `rem`, floor, bitwise ops, shifts, comparisons, boolean
  ops, `select`, and data-dependent array reads are carried as uninterpreted
  atoms, equal only when syntactically identical after canonicalizing their
  arguments. We lose completeness but not soundness.

## Requirements

The input to Volta is PTX code (the lowest level of the public-facing language stack for NVIDIA GPUs). PTX files can be generated from CUDA or CUTLASS code using `nvcc`.

We require that kernels are _structured-CTAs_. That is:

- Tensor/array sizes are statically known
- Branch targets and memory addresses can be resolved statically given the grid dimensions and input arrays
- There is no recursion

The only synchronization primitives we currently support are barriers, such as `syncwarp`, `syncthreads`, and the implicit warp-level barriers of tensor core operations (`mma.sync`, `wmma.*`, `ldmatrix`, `shfl.sync`). We do not support asynchronous primitives such as `arrive` and `wgmma`.

## Building

```bash
cargo build --release   # release mode matters: ~20x faster analysis
cargo test --workspace  # run the test suite
```

## Usage

### Parse a PTX file (syntax check)

```bash
cargo run --release -- parse <file.ptx>
```

### Analyze one kernel

Symbolically executes a kernel: reports data races and deadlocks, and prints
the symbolic expressions for each output array element.

```bash
cargo run --release -- analyze <file.ptx> -k <kernel> -b 32,4 -g 1 \
    --array "vals:0x100000000:4:2048:in" \
    --array "out:0x200000000:4:2048:out" \
    --param ptr:out --param ptr:vals --param int:2048 \
    --dyn-shared 1024
```

- `-k, --kernel`: kernel entry name (defaults to the first kernel in the module)
- `-b, --block` / `-g, --grid`: launch dimensions, e.g. `128` or `32,4,1`
- `--array "name:base:elem_width:len:kind"` (repeatable): declares a global
  array at address `base` with `len` elements of `elem_width` bytes; `kind`
  is `in` (symbolic input), `out`, `inout`, or `index` (concrete
  `arr[i] = i`, for index/permutation inputs)
- `--param` (repeatable, in declaration order): `int:N`, `float:X`,
  `sym:name` (a named symbolic float), or `ptr:array_name`
- `--global NAME=value` (repeatable): module-scope `.global` variable values
- `--dyn-shared N`: dynamic (extern) shared memory bytes
- `--print-outputs N`: print up to N elements per output array (default 8)
- `--no-profile`: skip the per-instruction-kind execution profile (shown by default)

### Compare two kernels

`volta compare` checks a reference/optimized pair for equivalence directly
from the CLI (races/deadlocks are still checked for each kernel individually).
Arrays/params/globals are shared by both kernels by default; give `--block2`/
`--grid2` if the optimized kernel's launch config differs (e.g. a tiled
kernel vs. a grid-stride reference).

```bash
cargo run --release -- compare <ref.ptx> <opt.ptx> \
    --kernel1 <ref_kernel> --kernel2 <opt_kernel> -b 128 \
    --array "in:0x10000:4:128:in" --array "out:0x20000:4:1:out" \
    --param ptr:in --param ptr:out
```

- `--footprint exact|intersect` (default `intersect`): how to pair up the
  two kernels' written output indices - `intersect` compares only the
  common indices (e.g. a grid-stride reference vs. a tiled kernel)
- `--sample N`, `--verify-numeric`, `--recycle-terms N`: same meaning as the
  `volta_bench` flags below
- `--no-profile`: skip the per-instruction-kind execution profile (shown by default)
- `--backend decision|z3` (default `decision`): which decision procedure to
  check equivalence with - see [Z3 backend](#z3-backend)

**VC dump/replay**: after symbolic execution, persist both kernels'
verification conditions (the expression arena + output footprint) to disk,
then rerun just the equivalence check from that dump later - no PTX parsing,
lowering, or symbolic execution involved on replay.

```bash
cargo run --release -- compare <ref.ptx> <opt.ptx> ... --dump-vcs pair.vcdump
cargo run --release -- compare --from-dump pair.vcdump   # rerun later, instantly
```

### Logging

Every `volta`/`volta-bench` run writes a log file under `volta-logs/`
(`<timestamp>-<command>.log`), recording the exact command line and a
one-line outcome summary - independent of the `logging` feature, so it
works in a plain build. Pass `--log-dir <path>` to change the directory or
`--no-log-file` to disable it. Building with `--features logging` also
mirrors the `log` crate's trace/debug/info/warn output into the same file
(`--log-level`), in addition to stderr:

```bash
cargo run --release --features logging -- --log-level info analyze ...
```

### Z3 backend

Checks the same verification conditions with Z3 (via SMT-LIB2, shelling out
to the `z3` CLI - no FFI/bindgen, just `z3` on `PATH`) instead of Volta's
own decision procedure, for a timing/capability comparison. This reproduces
the finding from on Volta's
independent kernel corpus: Z3 decides the polynomial fragment (matmul) and
is competitive, but returns `unknown` on the exponential fragment
(softmax/attention) at realistic sizes - not a bug, a fragment Z3's
decidable theories don't cover, which is exactly why the specialized
decision procedure exists.

```bash
sudo apt-get install -y z3   # only prerequisite - no libz3-dev/libclang-dev
cargo run --release -- compare <ref.ptx> <opt.ptx> ... --backend z3
```

Covers the arithmetic + `Exp` + `Max`/`Min` fragment (same atom-naming
convention as the decision procedure, so cross-kernel symbol correlation
matches); anything else (`Select`, comparisons, bitwise ops,
data-dependent array reads, ...) is reported `unsupported` for that element
rather than guessed at unsoundly. See `crates/volta_z3/src/translate.rs`
for the exact boundary.

### Reproduce the paper's evaluation

`volta_bench` runs every benchmark from the paper (39 in total) over the PTX
collected in `crates/volta_bench/kernels/`.

```bash
cargo run --release -p volta_bench -- list
cargo run --release -p volta_bench -- all
cargo run --release -p volta_bench -- category <reduction|matmul|attention|causal|conv|agent|tilelang|race>
cargo run --release -p volta_bench -- single "(Attention, FA1)"
```

Useful flags (global):

- `--sample N`: check at most N output elements per array (0 = all)
- `--verify-numeric`: confirm every verdict with the f64 oracle
- `--recycle-terms N`: recycle the VC intern tables past N interned terms
  (0 = never). Lower values bound memory at the cost of re-canonicalizing
  shared structure
- `--json <path>` (on `all`/`category`): export results as JSON

`single` also prints a per-instruction-kind execution profile for both
kernels automatically (matching `volta compare`'s default); `all`/`category`
stay compact and don't, to avoid flooding the table with one profile per
benchmark row.

To compare against Z3 instead of (or alongside) the decision procedure, use
`z3-compare` (needs `z3` on `PATH` - see [Z3 backend](#z3-backend)):

```bash
cargo run --release -p volta_bench -- z3-compare all --json results.json
cargo run --release -p volta_bench -- z3-compare reduction
cargo run --release -p volta_bench -- z3-compare "(Attention, FA1)"
```

For every equivalence benchmark matched by the selector (`all`, a category,
or an exact benchmark name), this runs *both* backends and prints exec/
decision/Z3 timing side by side, plus Z3's per-element equivalent/
not-equivalent/unknown/unsupported/error breakdown. `--z3-timeout N` bounds
each Z3 query in seconds (default 30, `0` = no limit); `--sample`/
`--recycle-terms` (global flags) apply to both backends. The default
`all`/`category`/`single` commands never invoke Z3 - `z3-compare` is opt-in.

**Memory note**: symbolic execution plus a warm VC session can use tens of
GiB on the attention benchmarks (each output row retains a large shared
softmax denominator). On machines with limited RAM, run one category at a
time and bound the VC tables, e.g.:

```bash
bash -c 'ulimit -v 12582912; exec cargo run --release -p volta_bench -- \
    --recycle-terms 250000 category attention'
```

which holds peak memory near the symbolic-execution floor (~5 GiB) in
exchange for slower VC checking. The other categories are far lighter
(full matmul: ~2 GiB).

## Citation

```bibtex
@misc{dubey2025equivalencecheckingmlgpu,
      title={Equivalence Checking of ML GPU Kernels},
      author={Kshitij Dubey and Benjamin Driscoll and Anjiang Wei and Neeraj Kayal and Rahul Sharma and Alex Aiken},
      year={2025},
      eprint={2511.12638},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2511.12638},
}
```

## License

This repository is licensed under [LICENSE](LICENSE). This implementation of Volta is completely independent from the Python implementation mentioned in the evaluation section of the arxiv paper, which was written by Kshitij Dubey and is owned by Microsoft Research. While I was a co-author of that paper, I never viewed Kshitij's implementation, nor did I discuss any details of it not presented in the arxiv paper.
