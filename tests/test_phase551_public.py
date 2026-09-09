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

class Phase5PublicExportTest(unittest.TestCase):
    def fixture(self):
        return {'format':'novllm-phase5-summary', 'candidate_count':1, 'freeze':False,
                'summaries':[{'candidate_id':'j-reversible-sp-unigram-32k','recipe':'J','vocab_size':32000,
                  'exact_round_trip_rate':1.0,'categories':{'kanbun':{'chars':100,'characters_per_token':1.5}}}],
                'shortlist':[],'shortlist_rationale':[]}

    def export_fixture(self, data, folder):
        import json
        path=folder/'input.json';path.write_text(json.dumps(data))
        return public.export_phase5(path,folder/'out')

    def test_allowlist_and_unknown_preservation(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=self.fixture()
            data['private_endpoint']='private-secret-marker'
            data['summaries'][0]['categories']['kanbun']['document_id']='private-secret-marker'
            data['summaries'][0]['categories']['kanbun']['text']='private-secret-marker'
            tables=self.export_fixture(data,root)
            self.assertEqual(tables['category_metrics'][0]['characters_per_token'],1.5)
            self.assertIsNone(tables['category_metrics'][0]['unknown_token_count'])
            self.assertNotIn('private-secret-marker',''.join(p.read_text() for p in (root/'out').iterdir()))
            metadata=json.loads((root/'out/provenance.json').read_text())
            self.assertEqual(metadata['source_sha256'],hashlib.sha256((root/'input.json').read_bytes()).hexdigest())

    def test_identity_and_type_injection_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for key,value in [('candidate_id','private-secret-marker'),('exact_round_trip_rate','private-secret-marker')]:
                data=self.fixture();data['summaries'][0][key]=value
                with self.assertRaises(ValueError):self.export_fixture(data,root)

    def test_lm_candidate_gate_not_expanded(self):
        fixture=PublicExportTest().run_fixture()
        fixture[0]['candidate']='j-reversible-sp-unigram-8k'
        with self.assertRaises(ValueError):public.aggregate([fixture])

if __name__ == '__main__':
    unittest.main()
