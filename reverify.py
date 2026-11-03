"""Reverify the paper's saved accepted trials without changing experiments.

Run: python -B -m official_neural_np4g.reverify
One resumable JSON result contains provenance, reports, and referenced ADF
proofs with their transitive dependencies. Workers enforce time/memory limits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
import traceback

from .runtime import configure_logging, default_artifact_root, default_output_root

ROOT = Path.cwd()
SUCCESS = "VERIFIED_EQUIVALENT"
logger = logging.getLogger(__name__)


def resolve_artifact_path(path, *, source_root=None, artifact_root=None):
    """Resolve recorded paths while preserving the bytes of historical ledgers.

    Older research ledgers recorded artifacts inside the source package. The
    optional artifact root maps that prefix to their current local location.
    Other relative paths are interpreted against the supplied source root.
    """
    recorded = Path(str(path).replace("\\", "/"))
    if recorded.is_absolute():
        return recorded.resolve()
    parts = recorded.parts
    if parts[:2] == ("official_neural_np4g", "artifacts"):
        tail = Path(*parts[2:])
        destination = (default_artifact_root() if artifact_root is None else
                       Path(artifact_root).expanduser().resolve())
        resolved = (destination / tail).resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError("recorded artifact path leaves its artifact root")
        return resolved
    base = ROOT if source_root is None else Path(source_root).expanduser().resolve()
    return (base / recorded).resolve()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replace_report(temporary: Path, output: Path):
    """An editor/virus scanner may briefly deny atomic replacement on Windows."""
    for attempt in range(31):
        try:
            os.replace(temporary, output)
            return
        except PermissionError:
            if attempt == 30:
                raise
            time.sleep(1)


def expression_from_key(key):
    """Decode the closed immutable IR grammar; never eval saved text."""
    from . import program_ir as ir
    tag, *a = key
    if tag == "const":
        return ir.Const(a[0])
    if tag in ("and", "or"):
        return (ir.And if tag == "and" else ir.Or)(*(expression_from_key(t) for t in a[0]))
    if tag == "not":
        return ir.Not(expression_from_key(a[0]))
    if tag == "rel":
        return ir.Rel(*a)
    if tag == "un":
        return ir.Un(*a)
    if tag == "unself":
        return ir.UnSelf(*a)
    if tag == "eq":
        return ir.Eq()
    if tag == "join":
        return ir.Join(*(expression_from_key(t) for t in a))
    if tag == "nodeat":
        return ir.NodeAt(expression_from_key(a[0]), a[1])
    if tag == "gexists":
        return ir.GExists(expression_from_key(a[0]))
    if tag in ("nexists", "nforall", "ncount"):
        r = a[0][1] if a[0][0] == "relname" else expression_from_key(a[0][1])
        arguments = [r, a[1], expression_from_key(a[2])]
        if tag == "ncount":
            arguments += [frozenset(a[3]), a[4]]
        return {"nexists": ir.NExists, "nforall": ir.NForall, "ncount": ir.NCount}[tag](*arguments)
    raise ValueError(f"unsupported saved expression tag {tag}")


def build_jobs(ledger, *, source_root=None, artifact_root=None):
    """Use exact frozen candidate hashes, including transitive ADF dependencies."""
    jobs = {}
    inputs = {}
    def tracked(path):
        path = Path(path)
        inputs[path.resolve().as_posix()] = digest(path)
        return read(path)
    def add_adf(cohort, record, index):
        key = "adf:" + record["task"] + ":" + record["checkpoint_sha256"]
        if key in jobs:
            return key
        dependencies = [add_adf(cohort, index[n], index) for n in record["dependencies"]]
        checkpoint = cohort / "adf_library" / record["checkpoint"]
        if digest(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"ADF hash mismatch: {checkpoint}")
        inputs[checkpoint.resolve().as_posix()] = record["checkpoint_sha256"]
        jobs[key] = {"key": key, "kind": "adf", "task": record["task"],
                     "domain": record["domain"], "seed": 0,
                     "cohort": cohort.resolve().as_posix(),
                     "checkpoint": checkpoint.resolve().as_posix(),
                     "checkpoint_sha256": record["checkpoint_sha256"],
                     "dependencies": dependencies, "dependency_names": record["dependencies"]}
        return key
    for task, groups in ledger["neural_np4g"].items():
        cohort_names = [name for name in ("reported", "initial", "additional") if name in groups]
        if not cohort_names:
            raise ValueError(f"no supported trial cohort for task {task}")
        for group in cohort_names:
            cohort_info = groups[group]
            cohort = resolve_artifact_path(cohort_info["origin"],
                                          source_root=source_root, artifact_root=artifact_root)
            accepted = [a for a in cohort_info["attempts"] if a["status"] == SUCCESS]
            if not accepted:
                continue
            index = {r["task"]: r for r in tracked(cohort / "adf_library/index.json")["entries"]}
            for attempt in accepted:
                directory = cohort / "runs" / task / f"attempt_{attempt['seed']:02d}"
                record = tracked(directory / "attempt_result.json")
                manifest = tracked(directory / "adf_candidate_manifest.json")
                if (record["status"] != SUCCESS or manifest["task"] != task
                        or manifest["attempt_seed"] != attempt["seed"]
                        or manifest["candidates"] != record["adf_candidates"]):
                    raise ValueError(f"trial provenance mismatch: {directory}")
                dependencies = []
                for candidate in manifest["candidates"]:
                    r = index[candidate["task"]]
                    if any(r[k] != candidate[k] for k in ("checkpoint", "checkpoint_sha256")):
                        raise ValueError(f"frozen ADF mismatch: {directory}")
                    dependencies.append(add_adf(cohort, r, index))
                checkpoint = directory / "stages" / record["selected_stage"] / "checkpoint.pt"
                sha = digest(checkpoint)
                if sha != record["checkpoint_sha256"] or sha != attempt["checkpoint_sha256"]:
                    raise ValueError(f"trial checkpoint hash mismatch: {checkpoint}")
                inputs[checkpoint.resolve().as_posix()] = sha
                key = f"trial:{group}:{task}:{attempt['seed']}"
                jobs[key] = {"key": key, "kind": "trial", "task": task,
                             "domain": record["domain"], "seed": attempt["seed"],
                             "cohort": cohort.resolve().as_posix(), "group": group,
                             "checkpoint": checkpoint.resolve().as_posix(),
                             "checkpoint_sha256": sha, "dependencies": dependencies,
                             "dependency_names": [c["task"] for c in manifest["candidates"]]}
    return jobs, inputs


def _worker(connection, job, upstream_reports):
    started = time.monotonic()
    try:
        import torch
        torch.set_num_threads(1)
        from .adf import ADFEntry, ADFLibrary
        from .model import NeuralNP4G
        from .train import load_checkpoint
        from .extract import extract_task
        from .config import derive_seed
        from .tasks import get_task
        from .extraction_worker import _json_value
        library = ADFLibrary(Path(job["cohort"]) / "adf_library")
        entries = library.load_entries(
            domain=job["domain"], device="cpu",
            model_factory=lambda r, deps: NeuralNP4G.from_config(r["build_config"], deps))
        by_name = {e.task: e for e in entries}
        dependencies = [by_name[n] for n in job["dependency_names"]]
        if digest(job["checkpoint"]) != job["checkpoint_sha256"]:
            raise ValueError("checkpoint changed since planning")
        if job["kind"] == "adf":
            entry = by_name[job["task"]]
            if entry.checkpoint_sha256 != job["checkpoint_sha256"]:
                raise ValueError("ADF entry changed since planning")
        else:
            model, _ = load_checkpoint(job["checkpoint"], dependencies, device="cpu")
            task = get_task(job["task"])
            entry = ADFEntry(task=task.name, domain=task.domain, arity=task.target_arity,
                             input_dims=tuple(model.input_dims), model=model,
                             dependencies=tuple(job["dependency_names"]),
                             checkpoint_sha256=job["checkpoint_sha256"])
        formulas = {}
        for dependency, key in zip(dependencies, job["dependencies"]):
            proof = upstream_reports[key]
            if proof["status"] != SUCCESS or proof["checkpoint_sha256"] != dependency.checkpoint_sha256:
                raise ValueError("dependency proof identity mismatch")
            formulas[dependency.task] = {"report": proof,
                                        "expr_raw": expression_from_key(proof["raw_expression"])}
        task = get_task(job["task"])
        sanity_seed = derive_seed(job["seed"], task.index, "test_data", 91)
        falsification_seed = derive_seed(job["seed"], task.index, "falsification_data", 91)
        report = extract_task(job["task"], entry, dependencies, formulas, do_sanity=True,
                              sanity_seed=sanity_seed, falsification_seed=falsification_seed)
        expression = report.pop("expr_raw_obj", None)
        if job["kind"] == "adf" and expression is not None and report["status"] == SUCCESS:
            report["raw_expression"] = expression._key
        report.update(checkpoint_sha256=job["checkpoint_sha256"],
                      sanity_seed=sanity_seed, falsification_seed=falsification_seed)
        result = _json_value(report)
    except BaseException as error:
        result = {"status": "EXTRACTION_FAILED", "reason": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    result["worker_seconds"] = round(time.monotonic() - started, 3)
    connection.send(result)
    connection.close()


def run(output: Path, workers: int = 2, time_seconds: float = 1800,
        memory_gb: float = 6, task_filter: str | None = None, restart: bool = False,
        *, ledger_path: Path | str | None = None, source_root: Path | str | None = None,
        artifact_root: Path | str | None = None):
    import psutil
    from . import validated_numerics as vn
    if not 1 <= workers <= 4 or time_seconds <= 0 or memory_gb <= 0:
        raise ValueError("positive limits and 1–4 workers are required")
    source_root = ROOT if source_root is None else Path(source_root).expanduser().resolve()
    ledger_path = (source_root / "docs/paper_results.json" if ledger_path is None
                   else Path(ledger_path).expanduser().resolve())
    jobs, inputs = build_jobs(read(ledger_path), source_root=source_root, artifact_root=artifact_root)
    inputs[ledger_path.as_posix()] = digest(ledger_path)
    if task_filter:
        wanted = {key for key, j in jobs.items() if j["task"] == task_filter}
        if not wanted:
            raise ValueError(f"no accepted circuit for task {task_filter}")
        def retain(key):
            wanted.add(key)
            for d in jobs[key]["dependencies"]:
                if d not in wanted:
                    retain(d)
        for key in list(wanted):
            retain(key)
        jobs = {k: j for k, j in jobs.items() if k in wanted}
    source_hashes = {"official_neural_np4g/" + p.name: digest(p)
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    fingerprint = hashlib.sha256(json.dumps({"source": source_hashes, "input": inputs,
                                             "contract": vn.metadata()}, sort_keys=True).encode()).hexdigest()
    output = output.resolve()
    original_roots = {Path(j["cohort"]) for j in jobs.values()}
    if any(output.is_relative_to(p) for p in original_roots):
        raise ValueError("output must not be inside an original experiment")
    previous = read(output) if output.exists() else None
    if previous is not None and not restart:
        state = previous
        if state.get("fingerprint") != fingerprint or set(state.get("jobs", {})) != set(jobs):
            raise ValueError("inputs/code changed; use --restart to preserve this run and begin afresh")
    else:
        state = {"schema": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                 "fingerprint": fingerprint, "python": platform.python_version(),
                 "verification_contract": vn.metadata(), "source_sha256": source_hashes,
                 "input_sha256": inputs, "jobs": jobs, "results": {}}
        if previous is not None:
            state["previous_runs"] = previous.pop("previous_runs", []) + [previous]
    state["resource_limits"] = {"workers": workers, "per_circuit_seconds": time_seconds,
                                "per_worker_memory_gb": memory_gb}
    results = state["results"]
    def save():
        state["summary"] = {
            kind: dict(Counter(results[k]["status"] for k, j in jobs.items()
                               if j["kind"] == kind and k in results))
            for kind in ("trial", "adf")}
        state["completed"] = len(results)
        state["total"] = len(jobs)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        replace_report(temporary, output)
    save()
    context = mp.get_context("spawn")
    active = {}
    try:
        while len(results) < len(jobs):
            for key, job in jobs.items():
                if key in results or key in active or len(active) >= workers:
                    continue
                if not all(d in results for d in job["dependencies"]):
                    continue
                failed = [d for d in job["dependencies"] if results[d]["status"] != SUCCESS]
                if failed:
                    results[key] = {"status": "BLOCKED_ON_UPSTREAM", "dependencies": failed}
                    logger.warning("%s/%s %s: BLOCKED_ON_UPSTREAM", len(results), len(jobs), key)
                    save()
                    continue
                receive, send = context.Pipe(duplex=False)
                process = context.Process(target=_worker, args=(send, job,
                    {d: results[d] for d in job["dependencies"]}))
                process.start()
                send.close()
                active[key] = (process, receive, time.monotonic())
                logger.info("Started %s", key)
            for key, (process, connection, started) in list(active.items()):
                result = None
                if connection.poll():
                    try:
                        result = connection.recv()
                    except EOFError:
                        result = {"status": "EXTRACTION_FAILED", "reason": "worker exited without a report"}
                elif not process.is_alive():
                    result = {"status": "EXTRACTION_FAILED", "reason": f"worker exit {process.exitcode}"}
                else:
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        continue  # collect the exit or pipe report next iteration
                    if time.monotonic() - started > time_seconds or rss > memory_gb * 2**30:
                        result = {"status": "RESOURCE_LIMIT", "reason": "time limit" if
                                  time.monotonic() - started > time_seconds else "memory limit"}
                if result is not None:
                    if process.is_alive():
                        process.join(timeout=.2)
                    if process.is_alive():
                        process.terminate()
                    process.join()
                    connection.close()
                    del active[key]
                    results[key] = result
                    logger.info("%s/%s %s: %s (%.1fs)", len(results), len(jobs), key,
                                result["status"], time.monotonic() - started)
                    save()
            if not active and len(results) < len(jobs):
                # The next iteration may have newly available dependents.
                if not any(k not in results and all(d in results for d in j["dependencies"])
                           for k, j in jobs.items()):
                    raise ValueError("unresolved dependency cycle")
            time.sleep(.1)
    finally:
        for process, connection, _ in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
            connection.close()
        save()
    changed = [name for name, sha in inputs.items() if digest(name) != sha]
    state["original_inputs_unchanged"] = not changed
    state["changed_inputs"] = changed
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    logger.info("Reverification summary: %s", json.dumps(state["summary"], sort_keys=True))
    if changed:
        raise RuntimeError("original inputs changed during revalidation")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output_root() / "verification_results.json")
    parser.add_argument("--ledger", type=Path, help="Research ledger (default: SOURCE_ROOT/docs/paper_results.json)")
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Base for non-artifact relative paths in the ledger (default: current directory)")
    parser.add_argument("--artifact-root", type=Path, default=default_artifact_root(),
                        help="Local artifact directory for historical source-package paths")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-seconds", type=float, default=1800)
    parser.add_argument("--memory-gb", type=float, default=6)
    parser.add_argument("--task", help="Restrict to this task and its transitive ADFs")
    parser.add_argument("--restart", action="store_true",
                        help="Preserve existing results as audit history and rerun every circuit")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.time_seconds <= 0 or args.memory_gb <= 0:
        parser.error("positive limits and 1–4 workers are required")
    configure_logging("reverify")
    try:
        run(args.output, args.workers, args.time_seconds, args.memory_gb, args.task, args.restart,
            ledger_path=args.ledger, source_root=args.source_root, artifact_root=args.artifact_root)
    except KeyboardInterrupt:
        logger.warning("Reverification interrupted; saved results can be resumed")
        raise
    except Exception:
        logger.exception("Reverification failed")
        raise


if __name__ == "__main__":
    main()
