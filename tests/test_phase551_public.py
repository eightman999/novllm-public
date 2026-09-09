"""Publication boundary and independently recomputable measurements."""
import copy
import hashlib
import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('export_public', Path(__file__).resolve().parents[1] / 'scripts/export_phase551_public.py')
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)

class PublicExportTest(unittest.TestCase):
    def run_fixture(self):
        score = dict(bits_per_byte=2.0, nll_sum=20 * math.log(2), source_bytes=10, source_chars=5, tokens=3, token_nll=None)
        return ({'candidate': 'j-reversible-sp-unigram-32k', 'seed': 1, 'complete': True, 'validation': {'overall': score}, 'training': {'source_chars': 5, 'source_bytes': 10, 'tokens': 3}, 'parameter_report': {'total': 10, 'embedding': 4, 'non_embedding': 6}}, {}, {}, hashlib.sha256(b'fixture').hexdigest())

    def test_private_unlisted_fields_never_exported(self):
        m,c,h,d = self.run_fixture()
        m['private_endpoint'] = 'https://secret.invalid'
        m['training']['documents'] = ['private full text']
        c['path'] = '/home/private/dataset'
        h['gpu'] = 'secret host name'
        import json
        serialized = json.dumps(public.aggregate([(m,c,h,d)]))
        self.assertNotIn('secret', serialized)
        self.assertNotIn('private', serialized)
        self.assertIsNone(public.aggregate([(m,c,h,d)])['hardware'][0]['gpu'])

    def test_type_injection_rejected(self):
        fixture = self.run_fixture()
        fixture[0]['training']['tokens'] = '/home/private/path'
        with self.assertRaises(ValueError):
            public.aggregate([fixture])

    def test_wrong_bpb_rejected(self):
        fixture = self.run_fixture()
        fixture[0]['validation']['overall']['bits_per_byte'] = 1
        with self.assertRaises(ValueError):
            public.aggregate([fixture])

    def test_missing_and_ratios(self):
        tables = public.aggregate([self.run_fixture()])
        self.assertIsNone(tables['summary'][0]['peak_vram_bytes'])
        self.assertEqual(tables['parameters'][0]['embedding_ratio'], .4)
        self.assertEqual(tables['compression'][0]['token_ratio_to_j32'], 1)
        self.assertIsNone(tables['compression'][0]['token_ratio_to_j48'])

    def test_checkpoint_training_position_distinct_from_eval_denominator(self):
        fixture = self.run_fixture()
        score = fixture[0]['validation']['overall']
        fixture[0]['learning_curves'] = [{'requested_budget_fraction': .2,
            'actual_budget_fraction': .2001, 'source_chars': 6003000,
            'validation': {'overall': score, 'domains': {'aa': score}}}]
        row = public.aggregate([fixture])['checkpoint_domains'][0]
        self.assertEqual(row['train_source_chars'], 6003000)
        self.assertEqual(row['source_chars'], 5)

    def test_duplicate_runs_rejected(self):
        with self.assertRaises(ValueError):
            public.aggregate([self.run_fixture(), self.run_fixture()])

if __name__ == '__main__':
    unittest.main()
