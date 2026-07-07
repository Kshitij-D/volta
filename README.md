# Volta

> ⚠️ This is not the code that was used for the paper. That code was written by Kshitij Dubey at MSR and they have not agreed to released it. Hopefully, this reconstruction is useful; it appears to work on all the benchmarks from the paper. I've been quite busy, so some parts were written almost entirely by Fable 5 and I did not have time to do extensive quality assurance. ⚠️

Volta is a data race and equivalence checker for NVIDIA GPU kernels, implementing the approach from ["Equivalence Checking of ML GPU Kernels"](https://arxiv.org/pdf/2511.12638). Given a reference kernel implementation and an optimized counterpart, Volta proves their semantic equivalence over the reals, i.e., that they produce identical outputs for all valid inputs modulo floating point error, thereby verifying the correctness of the optimized kernel.

## Features

- **Deadlock Detection**: Identify deadlocks arising from over synchronization
- **Data Race Detection**: Identify races arising from under synchronization
- **Equivalence Checking**: Verify that optimized GPU kernels are semantically equivalent to their reference implementations

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
