import importlib.util
from pathlib import Path
import math
import pytest
spec=importlib.util.spec_from_file_location('decision',Path(__file__).parents[1]/'scripts/analyze_phase551.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.mark.parametrize('delta,expected',[(-1.01,'A'),(-1.0,'A'),(-.999,'B'),(-.5,'B'),(-.499,'C'),(0,'C'),(.499,'C'),(.5,'D'),(.501,'D')])
def test_preregistered_boundaries(delta,expected):
 assert m.classify(delta).startswith(expected+':')
@pytest.mark.parametrize('delta',[math.nan,math.inf,-math.inf])
def test_nonfinite_not_classified(delta):
 with pytest.raises(ValueError):m.classify(delta)
def test_no_j64_is_not_a_result(tmp_path):
 for name in ('summary','parameters','hardware','conditions','domain_bpb','checkpoints','provenance'):(tmp_path/(name+'.json')).write_text('[]')
 with pytest.raises(ValueError,match='complete J32/J48/J64'):m.analyze(tmp_path)
 assert not (tmp_path/'decision.json').exists()

# Synthetic J64 values below exercise gates only, never represent a measurement.
import copy
import json

@pytest.fixture
def complete_cohort(tmp_path):
 root=Path(__file__).parents[1]/'results'
 source=root/'phase55_public'
 if not source.exists():source=root/'phase55'
 names=('summary','parameters','hardware','conditions','domain_bpb','checkpoints','provenance')
 for name in names:
  rows=[r for r in json.loads((source/(name+'.json')).read_text()) if r['seed']==1]
  extra=[copy.deepcopy(r) for r in rows if r['candidate'].endswith('48k')]
  for r in extra:
   r['candidate']='j-reversible-sp-unigram-64k'
   if name=='parameters':r['vocab_size']=64000
   if name=='provenance':r['tokenizer_sha256']='6'*64
  (tmp_path/(name+'.json')).write_text(json.dumps(rows+extra))
 rows=json.loads((source/'tokenizer_config.json').read_text())
 provenance=json.loads((tmp_path/'provenance.json').read_text())
 for r in rows:r['tokenizer_sha256']=next(p['tokenizer_sha256'] for p in provenance if p['candidate']==r['candidate'])
 (tmp_path/'tokenizer_config.json').write_text(json.dumps(rows))
 return tmp_path

def mutate(folder, table, operation):
 p=folder/(table+'.json');rows=json.loads(p.read_text())
 operation(rows)
 p.write_text(json.dumps(rows))

def j64(rows):return next(r for r in rows if r['candidate'].endswith('64k'))

def test_complete_cohort_and_dynamic_report(complete_cohort):
 # A different seed and GPU must be rendered from inputs, not hardcoded prose.
 for table in ('summary','parameters','hardware','conditions','domain_bpb','checkpoints','provenance'):
  def update(rows):
   for r in rows:
    r['seed']=7
    if table=='hardware':r['gpu']='Tesla P100-PCIE-16GB'
  mutate(complete_cohort,table,update)
 assert m.analyze(complete_cohort,7)['decision'].startswith('C:')
 assert 'seed 7, Tesla P100-PCIE-16GB' in (complete_cohort/'J64_REPORT.md').read_text()

@pytest.mark.parametrize('table,key,value',[
 ('parameters','vocab_size',48000),('hardware','dtype','float16'),
 ('conditions','scheduler_axis',None),('conditions','total_steps',1),
 ('conditions','regime',None),('conditions','learning_rate',None),
 ('provenance','tokenizer_sha256','0'*64),('provenance','records_sha256','0'*64),
 ('tokenizer_config','input_sha256','0'*64),('tokenizer_config','recipe','other'),
])
def test_noncomparable_rejected(complete_cohort,table,key,value):
 mutate(complete_cohort,table,lambda rows:j64(rows).__setitem__(key,value))
 with pytest.raises(ValueError):m.analyze(complete_cohort)
 assert not (complete_cohort/'decision.json').exists()

@pytest.mark.parametrize('table',['domain_bpb','checkpoints'])
@pytest.mark.parametrize('corruption',['empty','missing','duplicate'])
def test_evaluation_cohort_required(complete_cohort,table,corruption):
 def corrupt(rows):
  if corruption=='empty':rows.clear()
  elif corruption=='missing':rows.remove(j64(rows))
  else:rows.append(copy.deepcopy(j64(rows)))
 mutate(complete_cohort,table,corrupt)
 with pytest.raises(ValueError,match='missing or duplicate'):m.analyze(complete_cohort)
 assert not (complete_cohort/'decision.json').exists()

def test_preprocessing_change_rejected(complete_cohort):
 mutate(complete_cohort,'tokenizer_config',lambda rows:j64(rows)['trainer_args'].__setitem__('remove_extra_whitespaces',True))
 with pytest.raises(ValueError,match='preprocessing'):m.analyze(complete_cohort)
