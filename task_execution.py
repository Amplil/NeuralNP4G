"""Selected-task execution inside the official package, outside benchmark claims."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import OFFICIAL_CONFIG, OfficialConfig
from .tasks import TASK_BY_NAME, fixed_adf_candidate_allowlist, get_task

SELECTED_PROTOCOL = OFFICIAL_CONFIG.protocol_version + '-selected-v1'
# v2 enables aggregation candidates when a unary custom task omits preprocess.
EMPIRICAL_PROTOCOL = OFFICIAL_CONFIG.protocol_version + '-empirical-v2'


def dependency_order(name: str) -> tuple[str, ...]:
    """Include transitive certified candidate obligations, in stable order."""
    ordered: list[str] = []
    active: set[str] = set()

    def visit(current: str) -> None:
        if current in ordered:
            return
        if current in active:
            raise ValueError('cyclic task dependencies')
        active.add(current)
        task = get_task(current)
        for dependency in dict.fromkeys((*task.dependencies, *(fixed_adf_candidate_allowlist(task) or ()))):
            visit(dependency)
        active.remove(current)
        ordered.append(current)

    visit(name)
    return tuple(ordered)


def _initialise(root: Path, request: dict[str, Any], config: OfficialConfig,
                device: torch.device) -> None:
    from . import run_official_experiments as runner

    manifest = {
        'protocol_version': config.protocol_version,
        'mode': request['mode'],
        'benchmark_comparable': False,
        'request': request,
        'config': config.as_dict(),
        'source_hashes': runner._runtime_source_hashes(),
        'execution_contract': runner._execution_contract(device),
        'numerical_runtime': {
            'threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        },
    }
    # Normalize tuples exactly as written to disk before checking a resume.
    manifest = json.loads(runner._json_bytes(manifest))
    path = root / 'task_execution_manifest.json'
    if path.exists():
        if runner._read_json(path) != manifest:
            raise ValueError('output belongs to a different task, dataset, source, or execution environment; use a new output directory')
    else:
        unexpected = [p for p in root.iterdir() if p.name != 'runner.lock']
        if unexpected:
            raise ValueError('output directory is not empty and has no matching task execution manifest')
        runner._atomic_json(path, manifest)
        runner._atomic_json(root / 'source_hashes.json', manifest['source_hashes'])
        runner._atomic_json(root / 'environment.json', runner._environment_record(device))


def _prepare_output(root: Path) -> Path:
    root = root.resolve()
    # Refuse unrelated runs before creating even a lock file in them.
    if root.exists() and not (root / 'task_execution_manifest.json').exists():
        if any(path.name != "runner.lock" for path in root.iterdir()):
            raise ValueError('output directory is not empty; use a new isolated directory')
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_selected(name: str, seed: int, root: Path, device: torch.device,
                 *, config: OfficialConfig = OFFICIAL_CONFIG) -> dict[str, Any]:
    from . import run_official_experiments as runner

    if seed not in config.attempt_seeds:
        raise ValueError('registered task seed must be one of the official seeds 0..9')
    order = dependency_order(name)
    config = replace(config, protocol_version=SELECTED_PROTOCOL)
    root = _prepare_output(root)
    request = {'mode': 'selected', 'task': name, 'seed': seed, 'execution_order': list(order),
               'selection_policy': 'one selected seed per task including certified prerequisites'}
    with runner._runner_lock(root):
        _initialise(root, request, config, device)
        runner.configure_logging("task-execution", log_dir=root / "logs")
        outcomes = {}
        for current in order:
            if not runner._global_dependencies_are_certified(root, get_task(current)):
                outcomes[current] = {'status': 'BLOCKED_BY_UPSTREAM_ADF', 'verification_status': 'NOT_PROVED'}
                continue
            result = runner.run_attempt(current, seed, root, device, config=config)
            outcomes[current] = result
            summary = {'protocol_version': config.protocol_version, 'benchmark_comparable': False,
                       'requested_task': name, 'seed': seed, 'tasks': outcomes}
            runner._atomic_json(root / 'summary.json', summary)
            if result['status'] == 'INFRA_FAILURE':
                raise RuntimeError(result.get('reason', 'infrastructure failure; rerun the same command to resume'))
        summary = {'protocol_version': config.protocol_version, 'benchmark_comparable': False,
                   'requested_task': name, 'seed': seed, 'tasks': outcomes}
        runner._atomic_json(root / 'summary.json', summary)
        return summary


def run_custom(path: Path, seed: int, root: Path, device: torch.device,
               *, config: OfficialConfig = OFFICIAL_CONFIG) -> dict[str, Any]:
    from . import run_official_experiments as runner
    from .custom_data import load_custom_task
    from .evaluate import evaluate_split
    from .train import build_model, train_model, load_checkpoint

    config = replace(config, protocol_version=EMPIRICAL_PROTOCOL)
    custom = load_custom_task(path, seed, config.protocol_version)
    request = {'mode': 'empirical', 'task': asdict(custom.task), 'seed': seed,
               'dataset_sha256': custom.source_sha256,
               'splits': {name: split.manifest.as_dict() for name, split in custom.splits.items()},
               'proof_policy': 'labels only; formal equivalence not assessed; no ADF promotion',
               'checkpoint_policy': 'fixed phase2 endpoint; no held-out selection'}
    root = _prepare_output(root)
    with runner._runner_lock(root):
        _initialise(root, request, config, device)
        runner.configure_logging("task-execution", log_dir=root / "logs")
        runner._atomic_json(root / 'task_input.json', custom.document)
        result_path = root / 'summary.json'
        # Revalidate immutable checkpoints on resume instead of trusting a
        # mutable summary. Existing endpoints are loaded without training.
        output = root / 'runs' / custom.task.name / f'seed_{seed:02d}'
        model = build_model(custom.task, seed, config=config)
        train_model(model, custom.splits['train'], output, device=device, stage='phase1', config=config)
        training = train_model(model, custom.splits['train'], output, device=device, stage='phase2', config=config)
        # Report the immutable endpoint, not a mutable optimizer state.
        model, _ = load_checkpoint(training.checkpoint_path, device=device)
        evaluations = {name: evaluate_split(model, custom.splits[name], device=device)
                       for name in ('validation', 'test')}
        result = {'protocol_version': config.protocol_version, 'benchmark_comparable': False,
                  'task': custom.task.name, 'seed': seed, 'status': 'EMPIRICAL_COMPLETE',
                  'verification_status': 'NOT_PROVED', 'formal_target': None,
                  'reason': 'User labels do not establish universal equivalence; no proof or ADF promotion was attempted.',
                  'checkpoint_sha256': training.checkpoint_sha256,
                  'checkpoint_path': str(training.checkpoint_path),
                  'selected_stage': 'phase2', 'selected_steps': training.steps,
                  'evaluations': evaluations}
        runner._atomic_json(result_path, result)
        return result


def command(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description='Official NP4G selected task execution (no arguments runs the full benchmark).')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--task', choices=tuple(TASK_BY_NAME))
    group.add_argument('--task-file', type=Path, help='declarative JSON task with train/validation/test data')
    parser.add_argument('--seed', type=int, default=0, choices=range(10))
    parser.add_argument('--output', type=Path, required=True, help='new isolated output directory, or exact same-run resume')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    parser.add_argument('--validate-only', action='store_true', help='validate custom JSON without training or creating outputs')
    args = parser.parse_args(argv)
    if args.validate_only:
        if args.task_file is None:
            parser.error('--validate-only requires --task-file')
        from .custom_data import load_custom_task
        custom = load_custom_task(args.task_file, args.seed, EMPIRICAL_PROTOCOL)
        print(json.dumps({'task': custom.task.name, 'dataset_sha256': custom.source_sha256,
                          'preprocess': list(custom.task.preprocess),
                          'valid': True, 'verification_status': 'NOT_PROVED'}))
        return
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    result = (run_selected(args.task, args.seed, args.output, device) if args.task
              else run_custom(args.task_file, args.seed, args.output, device))
    print(json.dumps(result, indent=2))
