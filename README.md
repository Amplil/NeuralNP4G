# Neural NP4G

Neural NP4G is a PyTorch model for relational reasoning with a shared learned
circuit, discrete states, candidate-gated relational primitives, and reusable
certified predicates (automatically defined functions, or ADFs). The package
contains the model, program extraction and verification, deterministic data
generation, and the fixed research experiment runners.

The public model package is distributed under the [MIT License](LICENSE).

The model can be imported and initialized directly. Saved research results,
checkpoints, and experiment logs are not required to use it.

## Installation

Use Python 3.10 or newer. Python 3.12 is used in the experiment command examples.
From the root of this repository (the directory containing `pyproject.toml`),
create a virtual environment and install the package:

```shell
python -m venv .venv
# POSIX; on Windows use .venv\Scripts\activate
. .venv/bin/activate
python -m pip install .
```

Dependencies are PyTorch, NumPy, `gmpy2` (MPFR for numerical verification), and
`psutil` (experiment resource checks), as declared in the package configuration.
A CUDA-enabled PyTorch installation is optional for model use and recommended
for the full training protocol.

## Hardware and resource requirements for the official experiments

The small example below is **not** a capacity test for the full experiment.
`run_official_experiments` schedules 14 tasks (10 primary tasks and 4 auxiliary
proof obligations), each with seeds `0..9`. Each seed trains for 25,000 steps
and, if not accepted, up to 50,000 steps, followed by proof-selected reporting.
Required upstream ADF failures can block downstream tasks. The separate
`run_paper_experiments` command also runs dense NLM; the estimates here cover
**NP4G only**, not the full NLM comparison.

### Capacity planning, not a certified minimum

The code analysis and interrupted run documented here **do not establish a
validated minimum RAM or VRAM specification for all 14 tasks and all 10 seeds**.
Use these provisional CPU provisioning targets for one run without concurrent
memory-heavy workloads, then measure process and machine high-water marks. They are
engineering allowances, not a guarantee of completion or proof success.
GiB means `2**30` bytes; total installed RAM and available RAM are different.

| CPU execution target | Total host RAM | Available RAM before starting | Evidence level |
|---|---:|---:|---|
| Initial provisioning target | 16 GiB | At least 14 GiB | Unmeasured planning target, not a validated minimum |
| Preferred dedicated long-run host | 32 GiB or more | At least 24 GiB | Extra margin for runtime, ADFs, allocator peaks and other activity; not full-battery certified |

For evaluation, roughly 5.6 GiB of overlapping join arrays plus an **assumed**
4 GiB allowance for models, data, Python/PyTorch and allocator retention, plus
2 GiB of spare capacity, rounds up to about 12 GiB. The separate proof phase can
need a 6 GiB worker plus that assumed 4 GiB resident-parent allowance and its
3 GiB available-memory floor: about 13 GiB. The lower starting target rounds
this planning scenario up to 14 GiB. These are different phases, so their
estimates are not added together. The 4 GiB allowance is not a measured upper
bound. The 32/24 GiB tier adds margin because no complete-battery peak has been
measured. OS and background applications consume the remainder of total RAM;
sharing a 16 GiB machine does not imply 14 GiB is available. A container/VM's effective memory limit may be
lower than the host total and must also accommodate the run. Swap is not counted
as available physical RAM or as a substitute for these targets.

### Why a small checkpoint can need several GiB

[evaluate.py](evaluate.py), `evaluate_split`, evaluates every structure of one
test size in a single forward pass. The largest registered cell is four
structures of 500 objects ([config.py](config.py), `TEST_CELLS`). Binary joins
in [preprocess.py](preprocess.py), `_join_exists` and `_join_pair_exists`, form
a float32 intermediate of shape `[B, N, N, N]`:

```text
4 examples * 500 * 500 * 500 * 4 bytes = 2,000,000,000 bytes = 1.86 GiB
```

Out-of-place multiplication, subtraction and ReLU can have two such arrays
live in the first channel (about 3.73 GiB). On later channels, the previous
`contribution` remains live while the next expression is evaluated, permitting
three arrays (about **5.59 GiB**) before other allocations. This is static eager
allocation accounting, **not a measured whole-process peak or total RAM
requirement**. The `n=200`, 16-structure cell is about 0.477 GiB per cubic array.
Model states, retained data, core intermediates, allocator caching, ADFs and OS
activity need additional capacity. `_join` computes primitives before multiplying
their learned gates, so an OFF gate is not a memory-saving guarantee
([model.py](model.py)).

The training batch size of eight does not split evaluation cells. The NP4G
evaluator has no microbatch/chunk-size option. The no-argument canonical runner
uses the full protocol; selected-task options are described below. NLM's separate microbatch implementation does not apply
to NP4G. Do not change test sizes, seeds, numerical dtype or verification limits
to make a run fit and then label it the same protocol.

### CPU, CUDA and proof-worker limits

- CPU-only PyTorch is supported. The canonical command selects CUDA automatically
  when `torch.cuda.is_available()` is true; otherwise it selects CPU. It does not
  automatically select Apple MPS or provide a device flag. Use a sustained
  multi-core host; no minimum core count or full-run duration is established.
- With CUDA, neural training/evaluation allocations use device VRAM. The cubic
  array estimate still applies on that device; host RAM cannot replace missing
  VRAM. No full-battery VRAM minimum or GPU speedup has been measured here.
  Keep host-memory capacity for data and verification as well. Changing CPU/CUDA
  device or runtime libraries during a run violates its execution contract.
- Extraction and strict verification run in a separate **CPU** subprocess. Its
  fixed limits are 6 GiB process RSS, a 3 GiB system-available-memory floor, and
  1,800 seconds per circuit; dependency rechecks have the same individual limits.
  A 6 GiB worker needs room in addition to its parent and the 3 GiB reserve.
  These are **worker limits, not whole-run RAM requirements**. The watchdog is
  polled, not a kernel-enforced allocation cap.
- Extraction guards do **not** wrap the main training/held-out evaluation
  process. A resource-limit extraction report means proof was not established
  within the budget, not a counterexample or a successful proof. A process killed
  without a report is a separate infrastructure event.

### Check before launch or resume

Run this resource inspection in the intended Python environment (POSIX shell;
on Windows, run the Python body as a script). It does not start an experiment. Check again near large evaluations and avoid concurrent
memory-heavy jobs. A startup snapshot cannot guarantee later availability.

```shell
python - <<'PY'
import psutil
import torch
GiB = 2**30
ram = psutil.virtual_memory()
print(f"RAM total={ram.total / GiB:.2f} GiB available={ram.available / GiB:.2f} GiB")
print(f"CPU logical={psutil.cpu_count()} torch_threads={torch.get_num_threads()}")
print(f"PyTorch={torch.__version__} CUDA_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"CUDA device={torch.cuda.get_device_name()} VRAM free={free / GiB:.2f} GiB total={total / GiB:.2f} GiB")
PY
```

Also check the container/VM memory limit and current usage with its platform
controls (on Linux, cgroup `memory.max`/`memory.current` when exposed). Host-level
`psutil`/`free` output alone may not show the effective limit. Use a writable,
persistent output directory with room for checkpoints, logs and proof artifacts;
a complete-battery disk minimum has not been measured. Keep backups of completed
results rather than relying on an ephemeral machine's lifetime.

### Observed interruption and recovery boundary

On 2026-10-12, source `24b0f495c87722d36c39b1db492b5375f5a76120` ran on a
CPU-only Linux host reporting 9.73 GiB total RAM, 9 logical CPUs and no swap
(Python 3.12.14, PyTorch 2.14.1+cpu). `has_father` completed all 10 seeds with
strictly verified results. For `is_sister` seed 0, phase-two training reached
50,000 steps and strict verification saved **NOT_EQUIVALENT with a counterexample**.
Validation completed; the main process was killed before a test completion
record, returning exit 137. Available RAM was approximately 2.2 GiB at the last
pre-failure snapshot and 3.7 GiB after exit. These snapshots were not a peak-RSS
measurement. Exit 137 and the allocation analysis are consistent with memory
pressure, but no attributable kernel/cgroup OOM record was available, so **OOM
was not confirmed**. The exact test cell at the kill was not logged.

At that boundary, identical source/configuration/environment and the same output
directory allow the runner to reuse completed training checkpoints and proof
results. Unfinished reporting restarts from validation, then test; it does not
retrain the completed 50,000 steps. Do not blindly retry with insufficient
resources, overwrite results, change proof statuses or bypass hashes. A stale
`RUNNING` state file after a hard kill is not proof a process is still running;
verify process/exit status and resources first.

Documentation-only changes do not change this runner's source manifest:
`_runtime_source_hashes` hashes `official_neural_np4g/*.py`, not Markdown
([run_official_experiments.py](run_official_experiments.py)). Python-code,
configuration or execution-contract changes still invalidate same-run reuse.
The runner records its fixed protocol and resume requirements in each output
manifest. Historical research-workspace documents are not included in this
standalone distribution.

## Basic use

This example initializes an untrained unary model and evaluates two small
structures. `input_dims` lists the widths of nullary, unary, binary, and ternary
fact channels.

```python
import torch

from official_neural_np4g.data import GraphBatch
from official_neural_np4g.model import NeuralNP4G

batch_size, objects = 2, 4
batch = GraphBatch(
    input0=torch.zeros(batch_size, 1),
    input1=torch.zeros(batch_size, objects, 1),
    input2=torch.zeros(batch_size, objects, objects, 1),
    input3=torch.zeros(batch_size, objects, objects, objects, 0),
    target=torch.zeros(batch_size, objects),
    pair_mask=torch.ones(batch_size, objects, dtype=torch.bool),
)
model = NeuralNP4G(
    num_objects=objects,
    target_arity=1,
    input_dims=(1, 1, 1, 0),
    message_ticks=1,
    channels=4,
    hidden_nodes=12,
    circuit_ticks=2,
).eval()
with torch.no_grad():
    logits = model(batch)
assert logits.shape == (batch_size, objects)
```

Unary targets return `[batch, objects]` logits; binary targets return
`[batch, objects, objects]` logits. `GraphBatch.target` and `pair_mask` support
training and evaluation and are not used by the model's forward pass. The
current model uses nullary, unary, and binary inputs; the ternary tensor is a
batch compatibility field. See [MODEL.md](MODEL.md) for the architecture,
candidate gates, ADF reuse, and the scope of numerical and logical guarantees.

## Run one task or your own dataset

Use the same official entry point with an explicit task selection:

```shell
python -m official_neural_np4g.run_official_experiments --task has_father --seed 0 --output local/my-has-father
python -m official_neural_np4g.run_official_experiments --task connectivity_6 --seed 3 --output local/my-connectivity
```

`--seed` defaults to 0 and accepts 0–9. Required ADF tasks and their transitive
candidate dependencies run first with the same selected seed. A dependent task
is blocked if an upstream predicate cannot be certified; uncertified ADFs are
never substituted. These runs use the unchanged official training and strict
proof machinery, but are **selected runs, not the full benchmark or a ten-seed
proof-rate estimate**. Use `--device cpu` or `--device cuda` to select a device.
Without arguments, the original full 14-task, ten-seed benchmark is unchanged.

A selected run acquires every prerequisite with the requested seed. The paper's
downstream trials instead share a library built from the earliest certified
seed for each prerequisite across the complete seed grid. Even running this
selected command separately for all ten seeds generally uses different ADF
libraries and does not reproduce the paper's downstream proof-rate estimates.

For your own task, put its definition and labeled relational examples in one
JSON file. Start from [examples/custom_inverse.json](examples/custom_inverse.json):

```shell
python -m official_neural_np4g.run_official_experiments --task-file examples/custom_inverse.json --output local/my-custom --validate-only
python -m official_neural_np4g.run_official_experiments --task-file examples/custom_inverse.json --seed 0 --output local/my-custom
```

Validation reports the resolved `preprocess` candidates, performs no training,
and creates no output. The sample demonstrates
the file format; its few examples are not a meaningful research dataset. A real
run still uses the official 25,000 + 25,000 optimization steps and model recipe;
selecting one task does not make that training a quick installation check.

The JSON object has `schema_version: 1`, `task`, and `splits`:

- `task`: `name` (lowercase identifier), `domain` (`graph` or `family`),
  `target_arity` (1 or 2), optional `message_ticks` (default 1), and optional
  `preprocess` (names from the existing primitive registry). When omitted,
  unary tasks use `["cexists", "cforall", "satcount"]`; binary tasks use `[]`.
- `splits`: separate, nonempty `train`, `validation`, and `test` lists. Each
  example has `input1`, `input2`, `target`, and optionally `mask`.
- For M objects, `input1` is `[M, 1]`; `input2` is `[M, M, 2]` for graph
  (adjacency, identity) or `[M, M, 4]` for family (father, mother, son,
  daughter). Graph adjacency has zero diagonal and identity is the identity
  matrix. Family relations are supplied explicitly, not derived by the loader.
- Unary targets/masks are `[M]`; binary targets/masks are `[M, M]` with zero
  diagonal. Omitted masks evaluate all unary cells or all off-diagonal binary
  cells. Every example must contain at least one evaluated cell.
- All tensor entries must be Boolean or 0/1. Each example is one finite
  relational structure; examples may have different object counts. Nullary and
  unused ternary inputs are zero-filled. Unknown keys, malformed dimensions,
  nonfinite values, and identical relational inputs across splits are rejected.
  The loader cannot detect semantic leakage or prove real-world label quality.

Input limits are 16 MiB per JSON file, 256 objects per example, 10,000 examples,
10 million tensor values, and 1–32 message ticks. These bound parsing; they are
not a promise that a maximum-size training run fits every machine.

This first custom-data interface supports fixed graph/family channel layouts
and supervised unary/binary Boolean labels. It does not accept free-form text,
images, arbitrary tabular schemas, executable Python, custom ADF dependencies,
or a formula string. `preprocess` chooses approved operations, never code.

Unary aggregation candidates let the circuit use adjacency or family relations
alongside node features. An explicit list selects just those candidates; for
example, `["cexists"]` supports existential aggregation. An explicit `[]` for a
unary task disables every relational aggregation, so predictions ignore all
binary relations. The loader warns about this node-only configuration. Binary
tasks with `[]` still receive raw relations and their transposes, but have no
join candidates. These are available operations, not a promise that training
will acquire a particular rule.

Custom tasks use the official model, deterministic optimizer trajectory, and
TRAIN-only pruning guards. They always train to the fixed phase-2 endpoint;
validation/test only report that endpoint and never select it. Their result is
`EMPIRICAL_COMPLETE` with `verification_status: NOT_PROVED`, even at 100%
accuracy. Labels alone cannot establish universal equivalence. Formal
verification remains available for registered tasks with existing formal
target definitions; custom-label tasks never enter the certified ADF library.
Here `NOT_PROVED` means that formal proof was not attempted. Keep these
empirical results separate from inconclusive verifier outcomes and from the
paper's benchmark denominators. This interface is additional implementation
functionality, not new benchmark evidence or certified acquisition and reuse
from arbitrary labels.

Each explicit run requires its own `--output` directory. Repeating the exact
command resumes compatible artifacts. The manifest binds the request, seed,
source hashes, runtime environment, configuration, and custom data/split hashes.
Changed inputs or sources require a new output directory. Selected and empirical
runs have distinct protocol suffixes (`-selected-v1`, `-empirical-v2`) and refuse
unrelated existing artifacts. Do not point these commands at an existing full
benchmark, paper run, or an older source tree's outputs.

Empirical v2 changes the omitted unary preprocessing default. Start a new output
directory for runs made with these sources; v1 manifests and checkpoints keep
their original identities and are not migrated or relabeled. The registered
benchmark's task definitions and preprocessing are unchanged.

## Source and development layout

The Python source lives at the repository root and is installed as the
`official_neural_np4g` package. `examples/custom_inverse.json` is also included
in the installed package. For use outside a checkout, locate it with:

```shell
python -c "from importlib.resources import files; print(files('official_neural_np4g').joinpath('examples/custom_inverse.json'))"
```

Run the standalone smoke tests after installation:

```shell
python -B -m unittest discover -s tests -v
```

CI builds the source distribution and wheel, installs the wheel, and runs these
tests outside the checkout on Python 3.10 and 3.12. They check runner imports,
unary/binary model inference, and custom-data CLI validation without training.
The original research workspace's full regression suite and experiment protocol
documents are not included here. These smoke tests do not replace that suite,
full training, or strict verification of a learned circuit. No research results,
checkpoints, or benchmark evidence are shipped in this repository.

## Experiments and local output

The registered protocol is `neural-np4g-full-20260919`. From an environment with this package installed,
run the complete Neural NP4G and dense NLM comparison with:

```shell
python -B -m official_neural_np4g.run_paper_experiments
```

Run only Neural NP4G with
`python -B -m official_neural_np4g.run_official_experiments`. Both commands resume
their own saved progress and use the fixed task and seed schedule. The full
protocol is a substantial training workload; neither command is a basic
installation check.

Generated files stay outside the source package. The installed package uses the operating system's user state directory as its
default output base, with experiment files in `artifacts/<protocol>/`. Set `NP4G_OUTPUT_DIR` to choose a different local base:

```powershell
$env:NP4G_OUTPUT_DIR = "D:\np4g-output"
python -B -m official_neural_np4g.run_paper_experiments
```

The official and paper runners use standard Python logging and save rotating
`official-experiments.log` and `paper-experiments.log` files in
`artifacts/<protocol>/logs/`. Standalone NLM and reverification commands default
to the output base's `logs/` directory; extraction logs accompany its selected
run directory unless `--log-dir` overrides them. Structured metrics,
checkpoints, proof reports, and provenance manifests remain in the experiment
artifact directory so runs can be resumed and audited. Importing or evaluating
the model does not configure application logging or create experiment output.

Resume requires matching source and environment hashes. Preserved artifacts
from earlier source versions retain their original hashes and cannot resume
under changed release sources. For a fresh experiment with this release, choose
an unused directory through `NP4G_OUTPUT_DIR`; existing research artifacts remain
available for separate reverification.

`python -B -m official_neural_np4g.reverify` checks previously saved research
circuits when their ledger and artifacts are supplied. Those files are separate
inputs to reverification; they are not dependencies of the model package.
Run `python -m official_neural_np4g.reverify --help` for input and output options.
