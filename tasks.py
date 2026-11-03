"""Fixed task registry for the official Neural NP4G experiment suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .preprocess import PreprocessOp, lookup_ops, site_for_arity


Domain = str


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One preregistered target predicate.

    ``message_ticks`` is part of the target's known composition depth, not a
    searched hyperparameter.  ``preprocess`` names the ordered primitive
    candidates exposed to the circuit.  ``dependencies`` is the subset of
    learned ADF candidates whose structural and causal use is required for
    this staged target.  The official NLM-10 curriculum gives each task only
    its preregistered, semantically relevant ADF bank.
    """

    index: int
    name: str
    domain: Domain
    target_arity: int
    message_ticks: int
    formula: str
    preprocess: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    base_input_policy: str = "all"

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("task index must be non-negative")
        if not self.name:
            raise ValueError("task name must be non-empty")
        if self.domain not in ("graph", "family"):
            raise ValueError(f"unknown domain: {self.domain}")
        if self.target_arity not in (1, 2):
            raise ValueError("target_arity must be 1 or 2")
        if self.message_ticks <= 0:
            raise ValueError("message_ticks must be positive")
        if not self.formula:
            raise ValueError("formula must be non-empty")
        if self.name in self.dependencies:
            raise ValueError("a task cannot depend on itself")
        if self.base_input_policy not in ("all", "adf_only"):
            raise ValueError("base_input_policy must be 'all' or 'adf_only'")
        if self.base_input_policy == "adf_only" and not self.dependencies:
            raise ValueError("adf_only tasks must declare required ADF dependencies")
        lookup_ops(self.preprocess, site_for_arity(self.target_arity))

    @property
    def base_input_dims(self) -> tuple[int, int, int, int]:
        # family binary channels: father, mother, son, daughter
        # graph binary channels: adjacency, identity
        return (1, 1, 4, 1) if self.domain == "family" else (1, 1, 2, 1)

    @property
    def uses_learned_adf(self) -> bool:
        return bool(self.dependencies)

    @property
    def preprocess_ops(self) -> tuple[PreprocessOp, ...]:
        """Candidate-gated aggregations or joins paired with this task."""

        return lookup_ops(self.preprocess, site_for_arity(self.target_arity))


HUSBAND_REQUIRED_ADF_NAMES = ("is_father", "is_mother", "co_parent")
HUSBAND_OPTIONAL_ADF_NAMES = ("has_father", "is_grandparent")
HUSBAND_ADF_CANDIDATE_NAMES = (
    "has_father",
    "is_grandparent",
    *HUSBAND_REQUIRED_ADF_NAMES,
)

# Every formal NLM-10 task has a fixed candidate bank. Compound rows receive
# only a sound decomposition and its transitive support; standalone rows get
# an empty bank. The legacy co_parent fixture retains its old empty bank.
FIXED_ADF_CANDIDATE_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "has_father": (),
    "is_sister": (),
    "has_sister": ("is_sister",),
    "is_parent": (),
    "is_grandparent": ("is_parent",),
    "is_brother": (),
    "is_uncle": ("is_brother",),
    # is_uncle itself consumes is_brother, so that frozen transitive support
    # must precede it when the ADF is evaluated for is_mguncle.
    "is_mguncle": ("is_brother", "is_uncle"),
    "adjacent_to_red": (),
    "connectivity_2": (),
    "connectivity_4": ("connectivity_2",),
    "connectivity_6": ("connectivity_2", "connectivity_4"),
    "outdegree_1": (),
    "outdegree_2": (),
    "co_parent": (),
}


def fixed_adf_candidate_allowlist(
    task: str | TaskSpec,
) -> tuple[str, ...] | None:
    """Return a preregistered candidate allowlist, or ``None`` for a prefix."""

    return FIXED_ADF_CANDIDATE_ALLOWLISTS.get(get_task(task).name)


_ROWS: tuple[tuple[str, Domain, int, int, str, tuple[str, ...]], ...] = (
    # Original NLM Family Tree rows and their certified helper predicates.
    ("has_father", "family", 1, 1, "exists y: father(x,y)", ()),
    ("is_sister", "family", 2, 2, "sister(x,y)", ()),
    (
        "has_sister",
        "family",
        1,
        1,
        "exists y: is_sister(x,y)",
        ("is_sister",),
    ),
    ("is_parent", "family", 2, 1, "father(x,y) or mother(x,y)", ()),
    (
        "is_grandparent",
        "family",
        2,
        1,
        "exists z: is_parent(x,z) and is_parent(z,y)",
        ("is_parent",),
    ),
    ("is_brother", "family", 2, 2, "brother(x,y)", ()),
    (
        "is_uncle",
        "family",
        2,
        1,
        "exists z: parent(x,z) and is_brother(z,y)",
        ("is_brother",),
    ),
    (
        "is_mguncle",
        "family",
        2,
        1,
        "exists z: mother(x,z) and is_uncle(z,y)",
        ("is_uncle",),
    ),
    # Original NLM General Graph rows. connectivity_2 is an auxiliary proof
    # obligation used to make the longer reachability tasks compositional.
    ("adjacent_to_red", "graph", 1, 1, "exists y: adj(x,y) and red(y)", ()),
    ("connectivity_2", "graph", 2, 2, "path of length at most 2", ()),
    (
        "connectivity_4",
        "graph",
        2,
        1,
        "connectivity_2(x,y) or exists z: connectivity_2(x,z) and connectivity_2(z,y)",
        ("connectivity_2",),
    ),
    (
        "connectivity_6",
        "graph",
        2,
        1,
        "connectivity_4(x,y) or exists z: connectivity_4(x,z) and connectivity_2(z,y)",
        ("connectivity_2", "connectivity_4"),
    ),
    ("outdegree_1", "graph", 1, 1, "count y: adj(x,y) == 1", ()),
    ("outdegree_2", "graph", 1, 1, "count y: adj(x,y) == 2", ()),
    # Legacy dILP/husband rows retained outside the formal NLM-10 battery.
    ("inverse_edge", "graph", 2, 1, "adj(y,x)", ()),
    ("undirected_edge", "graph", 2, 1, "adj(x,y) or adj(y,x)", ()),
    (
        "graph_colouring",
        "graph",
        1,
        1,
        "exists y: adj(x,y) and (red(x) iff red(y))",
        (),
    ),
    ("is_father", "family", 1, 1, "exists c: father(c,x)", ()),
    ("is_mother", "family", 1, 1, "exists c: mother(c,x)", ()),
    (
        "co_parent",
        "family",
        2,
        2,
        "x != y and exists c: parent(c,x) and parent(c,y)",
        (),
    ),
    (
        "husband",
        "family",
        2,
        1,
        "exists c: father(c,x) and mother(c,y)",
        HUSBAND_REQUIRED_ADF_NAMES,
    ),
    # Mandatory learned-ADF chain.  The downstream task is intentionally one
    # tick: its four-hop target must be obtained by composing the certified
    # two-hop ADF rather than by four raw message-passing rounds.
    ("two_hop", "graph", 2, 2, "exists z: adj(x,z) and adj(z,y)", ()),
    (
        "four_hop_via_adf",
        "graph",
        2,
        1,
        "exists z: two_hop(x,z) and two_hop(z,y)",
        ("two_hop",),
    ),
)

# Task-tied circuit-input preprocessing.  Names are ops at the site implied by
# target arity.  Empty means the circuit sees only state and raw relations.
_PREPROCESS: dict[str, tuple[str, ...]] = {
    "has_father": ("cexists",),
    "is_sister": ("cexists", "cexistsp_r"),
    "has_sister": ("cexists",),
    "is_parent": (),
    "is_grandparent": ("cexists",),
    "is_brother": ("cexists", "cexistsp_r"),
    "is_uncle": ("cexistsp_r",),
    "is_mguncle": ("cexistsp_r",),
    "adjacent_to_red": ("cexists",),
    "connectivity_2": ("cexists",),
    "connectivity_4": ("cexists",),
    "connectivity_6": ("cexists", "cexistsp_r"),
    "outdegree_1": ("satcount",),
    "outdegree_2": ("satcount",),
    "inverse_edge": (),
    "undirected_edge": (),
    "graph_colouring": ("cexists",),
    "is_father": ("cexists",),
    "is_mother": ("cexists",),
    "co_parent": ("cexistsp_r",),
    # The final stage combines certified ADF channels.  It intentionally has
    # no raw join primitive with which to relearn the shared-child relation.
    "husband": (),
    "two_hop": ("cexists",),
    "four_hop_via_adf": ("cexists",),
}

# The final husband stage is an ADF-composition obligation.  Its certified ADF
# channels are computed from the raw family relations before the base channels
# are masked, preventing the final circuit from relearning or decorating the
# answer with raw father/mother literals.
_BASE_INPUT_POLICY: dict[str, str] = {
    name: (
        "adf_only"
        if name
        in {
            "has_sister",
            "is_grandparent",
            "connectivity_4",
            "connectivity_6",
            "husband",
        }
        else "all"
    )
    for name, *_ in _ROWS
}


TASKS = tuple(
    TaskSpec(
        index,
        name,
        domain,
        arity,
        ticks,
        formula,
        _PREPROCESS[name],
        dependencies,
        _BASE_INPUT_POLICY[name],
    )
    for index, (name, domain, arity, ticks, formula, dependencies) in enumerate(_ROWS)
)
TASK_BY_NAME = {task.name: task for task in TASKS}

PRIMARY_TASK_NAMES = (
    "has_father",
    "has_sister",
    "is_grandparent",
    "is_uncle",
    "is_mguncle",
    "adjacent_to_red",
    "connectivity_4",
    "connectivity_6",
    "outdegree_1",
    "outdegree_2",
)
AUXILIARY_TASK_NAMES = ("is_sister", "is_parent", "is_brother", "connectivity_2")
BOUNDARY_TASK_NAMES = (
    "inverse_edge",
    "undirected_edge",
    "graph_colouring",
    "is_father",
    "is_mother",
    "co_parent",
    "husband",
)
ADF_TASK_NAMES = ("two_hop", "four_hop_via_adf")

# Fixed method-level curriculum. Auxiliary predicates are proof obligations,
# not paper comparison rows. Each compound target consumes a certified helper
# or earlier primary result through the allowlist above.
CURRICULUM_EXECUTION_ORDER = (
    "has_father",
    "is_sister",
    "has_sister",
    "is_parent",
    "is_grandparent",
    "is_brother",
    "is_uncle",
    "is_mguncle",
    "adjacent_to_red",
    "connectivity_2",
    "connectivity_4",
    "connectivity_6",
    "outdegree_1",
    "outdegree_2",
)


def _validate_registry() -> None:
    if len(TASK_BY_NAME) != len(TASKS):
        raise RuntimeError("task registry contains duplicate names")
    if set(_PREPROCESS) != {task.name for task in TASKS}:
        raise RuntimeError("preprocess lists must name exactly the registered tasks")
    if set(_BASE_INPUT_POLICY) != {task.name for task in TASKS}:
        raise RuntimeError("base-input policies must name exactly the registered tasks")
    seen: set[str] = set()
    for task in TASKS:
        missing = [name for name in task.dependencies if name not in seen]
        if missing:
            raise RuntimeError(
                f"task '{task.name}' depends on tasks not preceding it: {missing}"
            )
        seen.add(task.name)


_validate_registry()


@dataclass(frozen=True, slots=True)
class EmpiricalTaskSpec(TaskSpec):
    """Explicit custom-label task; never a registered formal proof target."""

    def __post_init__(self) -> None:
        TaskSpec.__post_init__(self)
        if not self.name.startswith("custom_") or self.name in TASK_BY_NAME:
            raise ValueError("empirical task names must use the custom_ namespace")
        if self.index != len(TASKS):
            raise ValueError("empirical task index must follow the fixed registry")
        if self.dependencies or self.base_input_policy != "all":
            raise ValueError("empirical tasks cannot consume or promote certified ADFs")


def get_task(task: str | TaskSpec) -> TaskSpec:
    if isinstance(task, EmpiricalTaskSpec):
        return task
    if isinstance(task, TaskSpec):
        registered = TASK_BY_NAME.get(task.name)
        if registered != task:
            raise ValueError(f"unregistered TaskSpec: {task.name}")
        return task
    try:
        return TASK_BY_NAME[str(task)]
    except KeyError as error:
        raise ValueError(f"unknown task: {task}") from error


def input_dims_for(
    task: str | TaskSpec,
    learned_adfs: Iterable[Any] = (),
) -> tuple[int, int, int, int]:
    """Return input widths after appending the supplied certified ADF entries."""

    spec = get_task(task)
    d0, d1, d2, d3 = spec.base_input_dims
    entries = tuple(learned_adfs)
    if isinstance(spec, EmpiricalTaskSpec) and entries:
        raise ValueError("empirical tasks cannot consume certified ADFs")
    names = tuple(str(getattr(entry, "task", "")) for entry in entries)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("learned ADF candidates must have unique non-empty names")
    wrong_domain = [
        name
        for name, entry in zip(names, entries)
        if str(getattr(entry, "domain", "")) != spec.domain
    ]
    if wrong_domain:
        raise ValueError(
            f"task '{spec.name}' received cross-domain ADFs {wrong_domain}"
        )
    missing = [name for name in spec.dependencies if name not in names]
    if missing:
        raise ValueError(
            f"task '{spec.name}' is missing required ADFs {tuple(missing)}"
        )
    for entry in entries:
        arity = int(getattr(entry, "arity", 0))
        if arity == 1:
            d1 += 1
        elif arity == 2:
            d2 += 1
        else:
            raise ValueError("learned ADF entries must have arity 1 or 2")
    return d0, d1, d2, d3


__all__ = [
    "ADF_TASK_NAMES",
    "AUXILIARY_TASK_NAMES",
    "BOUNDARY_TASK_NAMES",
    "CURRICULUM_EXECUTION_ORDER",
    "EmpiricalTaskSpec",
    "FIXED_ADF_CANDIDATE_ALLOWLISTS",
    "HUSBAND_ADF_CANDIDATE_NAMES",
    "HUSBAND_OPTIONAL_ADF_NAMES",
    "HUSBAND_REQUIRED_ADF_NAMES",
    "PRIMARY_TASK_NAMES",
    "TASKS",
    "TASK_BY_NAME",
    "TaskSpec",
    "fixed_adf_candidate_allowlist",
    "get_task",
    "input_dims_for",
]
