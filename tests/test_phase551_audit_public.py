import importlib.util,json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('audit_export',Path(__file__).parents[1]/'scripts/export_phase551_audit_public.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_unknown_is_not_an_invented_identity():
 assert m.opaque('unknown')=='unknown'
 assert m.opaque(None) is None
 assert m.opaque('private-id')==m.opaque('private-id')
 assert 'private-id' not in m.opaque('private-id')

def fixture(tmp):
 src=tmp/'input';src.mkdir()
 r={'eval_item_id':'PRIVATE_ITEM','source_work_identity':'PRIVATE_WORK','matched_train_id':'PRIVATE_TRAIN','category':'kanbun','max_similarity':.9,'threshold':.8,'status':'match_detected','work_overlap':'unknown','text':'UNPUBLISHABLE_TEXT_SAMPLE',**{k:False for k in m.FLAGS}}
 for scope in m.SCOPES:(src/(scope+'_audit.jsonl')).write_text(json.dumps({**r,'comparison_scope':scope})+'\n')
 (src/'eval_audit.json').write_text(json.dumps({'old':{},'candidates':{},'hardened':{},'excluded_work_count':1,'flagged_eval_item_count':1,'eval_internal_near_duplicate_pairs':0,'hardened_eval_sha256':'a'*64,'lm_dataset_sha256':'b'*64,'scopes':[],'inputs':[]}))
 (src/'hardened_manifest.jsonl').write_text('')
 (src/'eval_internal_near_duplicate.jsonl').write_text('')
 return src

def test_extra_text_and_private_ids_never_published(tmp_path):
 src=fixture(tmp_path);out=tmp_path/'out';m.export(src,out)
 serialized=''.join(p.read_text() for p in out.iterdir())
 for private in ('UNPUBLISHABLE_TEXT_SAMPLE','PRIVATE_ITEM','PRIVATE_WORK','PRIVATE_TRAIN'):assert private not in serialized
 assert 'unknown' in serialized

def test_threshold_change_rejected(tmp_path):
 src=fixture(tmp_path)
 p=src/'lm_train_30m_audit.jsonl';r=json.loads(p.read_text());r['threshold']=.95;p.write_text(json.dumps(r)+'\n')
 with pytest.raises(ValueError,match='threshold'):m.export(src,tmp_path/'out')
