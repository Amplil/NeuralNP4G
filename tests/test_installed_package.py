"""Small installed-distribution checks; these do not certify benchmark results."""
import importlib
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from official_neural_np4g.data import GraphBatch
from official_neural_np4g.model import NeuralNP4G


class InstalledPackageTests(unittest.TestCase):
    def test_public_modules_import(self):
        # CLI modules also load extraction, verification and optional runner dependencies.
        for name in ('run_official_experiments', 'run_paper_experiments',
                     'extraction_worker', 'reverify', 'nlm_baseline'):
            with self.subTest(module=name):
                importlib.import_module('official_neural_np4g.' + name)

    def test_forward_unary_and_binary(self):
        for arity in (1, 2):
            with self.subTest(arity=arity):
                shape = (2, 4) if arity == 1 else (2, 4, 4)
                batch = GraphBatch(
                    input0=torch.zeros(2, 1), input1=torch.zeros(2, 4, 1),
                    input2=torch.zeros(2, 4, 4, 1), input3=torch.zeros(2, 4, 4, 4, 0),
                    target=torch.zeros(shape), pair_mask=torch.ones(shape, dtype=torch.bool),
                )
                model = NeuralNP4G(num_objects=4, target_arity=arity,
                                   input_dims=(1, 1, 1, 0), message_ticks=1,
                                   channels=4, hidden_nodes=12, circuit_ticks=2).eval()
                with torch.no_grad():
                    logits = model(batch)
                self.assertEqual(tuple(logits.shape), shape)
                self.assertTrue(torch.isfinite(logits).all())

    def test_shipped_custom_example_validates_without_output(self):
        example = files('official_neural_np4g').joinpath('examples/custom_inverse.json')
        self.assertTrue(example.is_file())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            result = subprocess.run(
                [sys.executable, '-m', 'official_neural_np4g.run_official_experiments',
                 '--task-file', str(example), '--output', str(output), '--validate-only'],
                cwd=directory, capture_output=True, text=True, timeout=60, check=True,
            )
            report = json.loads(result.stdout)
            self.assertTrue(report['valid'])
            self.assertEqual(report['verification_status'], 'NOT_PROVED')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
