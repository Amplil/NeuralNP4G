# Neural NP4G model

`NeuralNP4G` is the model implementation used by the registered research
experiments. The model can also be constructed directly in Python; the fixed
experiment configuration below describes the research protocol, rather than
every possible constructor argument.

The current protocol is `neural-np4g-full-20260919`. Its primary comparison uses
the ten relational reasoning tasks from the NLM paper. Four helper predicates
are additional proof obligations for staged composition. Helpers are excluded
from the primary comparison denominator, but a downstream task cannot proceed
until its required helpers are certified.

## 1. Fixed research configuration

The official runners execute one registered method:

- Four state channels, twelve hidden nodes, and two internal circuit ticks.
  The task-specific number of message ticks is fixed in `tasks.py` from the
  known composition depth and is not searched.
- Each message tick reuses the same free circuit, `NP4GCircuit`. Input nodes
  are fixed, hidden nodes use sigmoid, and output nodes are linear.
- State updates use discrete `step` semantics. Evaluation and immutable proof
  checkpoints use exact `{-1,+1}` states. During training only, a fixed
  homotopy moves soft values to hard sign values while retaining the tanh
  surrogate gradient.
- The homotopy coefficient increases linearly to one during the first 40% of
  the complete 50,000-step schedule. Every attempt uses this schedule.
- Circuit edges, hidden nodes, primitive candidates, and learned ADF
  candidates use exact 0/1 forward gates and sigmoid gradients through a
  straight-through estimator (STE).
- Binary models use separate unary embeddings for the source and target.
  This allows directed unary ADFs, such as `is_father(x)` and `is_mother(y)`,
  to form asymmetric binary predicates.
- Structural loss, candidate sparsity, gate binarization, state saturation,
  scheduled fan-in pruning, and phase-two synthetic saturation and repair
  pruning train sparse discrete circuits suitable for extraction.

For unary tasks, candidate aggregations operate in both directions of each
binary relation. The free circuit receives the current state and its global
maximum alongside these aggregates. For binary tasks, inputs include the
current pair state, its transpose, candidate joins, raw relations, and their
transposes. The readout returns one logit per object or ordered object pair.

## 2. Unified primitive and learned ADF candidate gates

`TaskSpec.preprocess` specifies the aggregation or join candidates presented to
a task. Every candidate has a hard 0/1 STE gate in an ordered
`CandidateGateBank`. The bank contains:

- Primitive candidates registered for the task.
- Learned ADF candidates from previously certified tasks in the same domain
  whose status is `VERIFIED_EQUIVALENT`.

The available primitives are defined in `preprocess.py`:

| Site | Operation | Meaning |
|---|---|---|
| Unary aggregation | `cexists` | A true object exists along the relation |
| Unary aggregation | `cforall` | All objects along the relation are true |
| Unary aggregation | `satcount` | Saturating count of true related objects |
| Binary join | `cexists` | Existential join through an intermediate object using the same channel |
| Binary join | `cexistsp_r` | Existential join that distinguishes channel pairs |

Primitives operate on the current state at every tick; they are not static
data-generation transforms. An OFF candidate zeros its output block. Raw input
facts are not themselves primitive candidates.

Candidate sparsity is normalized to a fixed reference width of five
candidates, matching the previously certified recipe. A task-specific bank
with one candidate does not receive five times the OFF pressure on that
required candidate. Hard gates can cross their threshold during training, so
a temporary OFF gate never permanently prunes its circuit input columns.

A learned ADF freezes its certified upstream model and computes a hard 0/1
predicate once. Unary results are appended to `input1`, and binary results to
`input2`. Binary ADF diagonals are zeroed. Gradients do not flow from downstream
models into upstream models, and ADFs are not dynamically invoked inside ticks.

Each downstream attempt receives only its preregistered, semantically relevant
ADF bank. Transitive dependencies required to evaluate a composite ADF are
included in order; unrelated earlier ADFs are excluded. Candidate task names,
checkpoints, SHA-256 hashes, and order are fixed in
`adf_candidate_manifest.json` and cannot change on resume.
`TaskSpec.dependencies` identifies the subset whose actual use is required
for downstream acceptance.

## 3. Staged ADF curriculum

The current experiment executes tasks in this fixed order:

```text
has_father,
is_sister, has_sister,
is_parent, is_grandparent,
is_brother, is_uncle, is_mguncle,
adjacent_to_red,
connectivity_2, connectivity_4, connectivity_6,
outdegree_1, outdegree_2
```

After strict verification succeeds, the task's frozen circuit and extracted
formula are promoted to the same protocol's `ADFLibrary`. Checkpoints and proof
receipts from another protocol cannot become ADFs. All upstream ADFs are
trained, extracted, and verified under the same artifact root.

Family relations are stored in the `child -> parent` direction. The staged
Family Tree definitions are:

```text
is_sister(x,y)     := sister(x,y)
has_sister(x)      := EXISTS y: is_sister(x,y)
is_parent(x,y)     := father(x,y) OR mother(x,y)
is_grandparent(x,y):= EXISTS z: is_parent(x,z) AND is_parent(z,y)
is_uncle(x,y)      := EXISTS z: parent(x,z) AND is_brother(z,y)
is_mguncle(x,y)    := EXISTS z: mother(x,z) AND is_uncle(z,y)
```

General Graph reachability uses these stages, where `o` denotes relational
composition:

```text
connectivity_2 := path length <= 2
connectivity_4 := connectivity_2 OR connectivity_2 o connectivity_2
connectivity_6 := connectivity_4 OR connectivity_4 o connectivity_2
```

For `has_sister`, `is_grandparent`, `connectivity_4`, and `connectivity_6`, the
ADF inputs can construct the complete target. After computing upstream ADFs
from raw facts, the downstream model's raw unary and binary channels are
masked to zero. `is_uncle` composes certified `is_brother` with raw `parent`.
`is_mguncle` retains raw inputs for the final composition with `mother`, but
uses one message tick and requires certified `is_uncle`.

The bank for `is_mguncle` places `is_brother` before `is_uncle` so that the
frozen `is_uncle` model can run. `is_brother` is transitive support; the final
task's directly required atom is `is_uncle`. Both `connectivity_2` and
`connectivity_4` are directly required by `connectivity_6`.

Neither `outdegree_1` nor `outdegree_2` can be derived from the other, so they
are independent tasks. Staged learning claims apply only where a semantic
decomposition exists.

Equivalence between the final formula and the target alone does not establish
staged reuse. A stage with required ADFs must also satisfy all of the following:

1. Every required ADF atom occurs in the extracted formula before substitution.
2. Every required ADF's hard gate is ON.
3. Disabling each required ADF individually on the independent `adf_causal`
   split introduces additional errors.

Failure of any condition produces `ADF_NOT_CAUSALLY_USED`, even if the closed
formula passes equivalence verification.

## 4. Proof-preserving pruning and phase-two repair

Ordinary `prune_step` removes structurally dead inputs, reduces each row to its
scheduled fan-in, and limits count inputs to at most two per node. A hard OFF
primitive or ADF gate does not make its input structurally dead.

Every scheduled pruning event is transactional. If the post-optimizer event
batch has zero errors, pruning must preserve zero errors. A fixed TRAIN guard
uses sixteen examples per training size from the independent `prune_guard`
stream (stream ID 9); pruning must not increase the baseline error count at
any size.

Once the homotopy reaches fully hard `step_alpha=1`, a model with zero errors
on all 256 materialized TRAIN examples per size is saved as an exact incumbent,
including its optimizer, RNG, and `step_alpha`. At subsequent scheduled
boundaries, an optimizer update or pruning event that introduces errors
restores the complete incumbent, including hardness. Incumbent saving and the
full TRAIN guard are disabled during the soft period. Neither guard uses
validation or test data.

Repair generates 512 fixed synthetic abstract inputs and directly evaluates
removing each surviving edge from rows with large violations. Only edges that
reduce violations by at least 75% are permanently removed through
`prune_mask`; at most four rows are handled per event. Repair does not change
weights, logits, or training data.

If the full deletion proposal violates the guards, at most 128 deterministic
partial proposals are evaluated, and only a safe subset of deletions is
committed. If no deletion is safe, the complete `prune_mask` is restored.
These rules and bounds are fixed before observing results. Training state and
metrics retain the guard seed, pre-pruning baseline, proposed and accepted
deletions, rollback reasons, RNG state, `repair_kills`, and `repair_rows`.
Resuming a stage follows the same trajectory. Accuracy and loss are
recomputed from the state actually accepted after optimization and pruning;
pre-update values are recorded in separate fields.

At scheduled phase-two pruning boundaries, a model with zero errors on all
TRAIN data and maximum fan-in at most six is frozen as an intermediate
compression checkpoint. At most four such checkpoints are proved in
`(max fan-in, mean fan-in, step)` order, followed by the phase-two boundary.
Validation and test do not determine that order.

## 5. Excluded mechanisms

The registered method excludes:

- Dale-style source-node signs and inhibitory nodes.
- Temperature parameters and temperature annealing.
- Public modes that switch state updates, ADF invocation, or candidate selection.
- Dynamic ADF calls with derived relations as arguments inside ticks.
- Validation/test-driven reinitialization, added data, hyperparameter selection,
  or checkpoint selection.

The strict acceptance result at the preregistered phase-one checkpoint is the
only signal determining whether an attempt stops or resumes into phase two.

## 6. Implementation map

| File | Role |
|---|---|
| `preprocess.py` | Candidate primitive aggregation and join operations |
| `circuit.py` | Shared free circuit with hard STE structural gates and permanent prune masks |
| `model.py` | Discrete step model, unified candidate bank, ordered binary embeddings, and repair pruning |
| `adf.py` | Hard candidate gates, static learned ADF reuse, and certified ADF storage |
| `program_ir.py` / `extract.py` | Discrete program representation, ADF substitution, extraction, and equivalence verification |
| `validated_numerics.py` | Outward-rounded interval arithmetic for the real-number semantics of saved parameters |
| `reverify.py` | Saved-circuit and dependency reverification, hash checking, and report persistence |
| `config.py` / `tasks.py` | Fixed NLM-10 protocol, primitives, staged curriculum, required ADFs, and ADF-only inputs |
| `run_paper_experiments.py` / `paper_evidence.py` | Complete experiment entry point and evidence bundle produced after completion |

See the external [experiment protocol](../docs/official_neural_np4g/EXPERIMENTS.md)
for phase checkpoints, training, evaluation, and aggregation.

## 7. Numerical and logical verification contract

Verification interprets saved finite binary parameters as exact real numbers.
It does not assert that PyTorch/CUDA floating-point execution has identical
outputs on all inputs. The correspondence with floating-point execution is
also checked on independent finite datasets, and any mismatch revokes
acceptance.

Numerical extraction applies the following enclosure rules from initialization
through readout:

- Each product and sum is evaluated as a separate float64 operation and widened
  outward immediately with `nextafter`. Widening only the result of a matrix
  product is insufficient. Exact operations, including addition of zero,
  multiplication by zero or `+/-1`, and cancellation, retain their exact result.
  The implementation assumes ordinary IEEE binary64 operations and gradual
  underflow.
- Sigmoid and tanh are enclosed using directed MPFR rounding. Sigmoid uses a
  nonpositive exponent argument and preserves the direction of lower and upper
  bounds through exponentiation, the denominator, and division.
- Interval endpoints are not shortened by rounding. Merging uses an envelope
  containing every original interval, and containment in the assigned interval
  is checked. Truth decisions use no epsilon; intervals crossing a threshold
  remain unresolved.
- Existential, universal, and composition operations distinguish zero from
  positive contributions with intervals. If `alpha` is a lower bound on a
  positive contribution, a saturating sum merges its tail only from a count
  `k` for which outward-rounded arithmetic establishes `k * alpha >= 8`.
  Otherwise an abstract envelope is retained.
- Only combinations logically established as unreachable are excluded.
  Unsupported operations, nonfinite numbers, and resource limits are not
  converted into successful verification.

The logical conjunctive-query (CQ) grammar permits conjunctions of positive
binary atoms, positive or negative basic unary atoms, equalities, and
inequalities. A union of conjunctive queries (UCQ) is a finite disjunction of
these queries. Verification enumerates every variable identification, preserves
explicit unary values, and completes every unspecified unary value with both
truth assignments. Structures violating constraints are excluded without
rewriting them. Counterexamples are rechecked against the containing source
query. Arbitrary callbacks are not accepted as constraints. General constraints
are universal prohibitions of conjunctions in this grammar.

Experiment constraints cover parent uniqueness, typing, irreflexivity,
exclusivity, and acyclicity, as well as the prohibition of graph self-edges.
These constraints are preserved by deleting objects and binary edges, so
enumerating finite structures constructed from a source-query witness is
sufficient.

The fast path accepts containment only when it finds a homomorphism under
which each target-query atom is implied by source-query atoms. Otherwise it
falls back to complete enumeration. Simplifying a local formula to a constant
also requires checking containment of the formula, or its negation, in false.
An expression outside this grammar, such as a negated binary atom, is not
simplified through this check.

Degree comparisons calculate their bound from every atom's saturation point
and the target degree. The final row represents all degrees at or above that
bound. The verifier does not truncate at a fixed degree of 32.

Algorithmic soundness and verification of the implementation or numerical
libraries in a proof assistant are distinct claims. Regression tests include
rational checks of affine arithmetic, comparisons with high-precision MPFR,
subnormal values, thresholds, constrained counterexamples, exhaustive small
structures, a degree-33 counterexample, and extraction of actual circuits.
Tests do not replace a proof over all inputs.
