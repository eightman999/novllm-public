#!/usr/bin/env python3
"""Read-only final-checkpoint reevaluation using the unchanged Phase55 scorer."""
import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from canonical_corpus import phase55_same_source_runtime as rt
from canonical_corpus.probe_lm import ProbeLM,LMConfig
from canonical_corpus.tokenizer_adapters import ReversibleSentencePieceAdapter


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--eval-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available()
    assert 'P100' in torch.cuda.get_device_name(0), 'this evaluator is restricted to the P100'
    rt.set_seed(20260909,threads=2)
    protocol=json.loads((a.bundle/'protocol.json').read_text())
    datasets={}
    for name,filename in [('old','old_eval_work_joined.jsonl'),('hardened','hardened_eval.jsonl')]:
        datasets[name]=[json.loads(x) for x in (a.eval_dir/filename).open(encoding='utf-8')]
        datasets[name]=[r for r in datasets[name] if r['category'] in ('kanbun','kakikudashi')]
    for seed in (1,2):
        for size in ('32k','48k'):
            candidate='j-reversible-sp-unigram-'+size
            path=a.bundle/'results/cuda_same_source/runs'/f'seed{seed}'/candidate/'artifacts/checkpoint.pt'
            digest=rt._file_hash(path)
            assert digest==path.with_suffix('.sha256').read_text().strip()
            payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            cfg=LMConfig(**payload['model_config']); model=ProbeLM(cfg).cuda()
            model.load_state_dict(payload['model']);model.eval()
            tokenizer_path=a.bundle/protocol['tokenizers'][candidate]['path']
            assert rt._file_hash(tokenizer_path)==protocol['tokenizers'][candidate]['sha256']
            tokenizer=ReversibleSentencePieceAdapter(tokenizer_path)
            for name,records in datasets.items():
                start=time.monotonic();prepared,checks=rt._prepare(tokenizer,records)
                items=[];accum=defaultdict(lambda:[0.,0,0,0]);work_accum=defaultdict(lambda:[0.,0,0,0])
                for r in prepared:
                    measured=rt.evaluate(model,[r],cfg,torch.device('cuda:0'),4096)['overall']
                    item={'eval_item_id':r['id'],'work_id':r['document_id'],'category':r['category'],**measured}
                    items.append(item)
                    vals=[measured[k] for k in ('nll_sum','tokens','source_chars','source_bytes')]
                    for bucket in (accum[r['category']],work_accum[(r['category'],r['document_id'])]):
                        for i,value in enumerate(vals):bucket[i]+=value
                result={'candidate':candidate,'seed':seed,'dataset':name,'checkpoint_sha256':digest,
                    'dataset_sha256':rt._file_hash(a.eval_dir/('hardened_eval.jsonl' if name=='hardened' else 'old_eval_work_joined.jsonl')),
                    'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
                    'evaluation_seconds':time.monotonic()-start,'preservation':checks,
                    'segmentation':'original lines; unchanged Phase55 evaluate(); BOS per line',
                    'domains':{k:rt._normalized(*v) for k,v in accum.items()},
                    'works':[{'category':k[0],'work_id':k[1],**rt._normalized(*v)} for k,v in work_accum.items()]}
                result['checkpoint_unchanged_after_evaluation'] = rt._file_hash(path) == digest
                assert result['checkpoint_unchanged_after_evaluation']
                result['runtime_code_sha256'] = {str(p.relative_to(ROOT)):rt._file_hash(p) for p in
                    (Path(rt.__file__), ROOT/'canonical_corpus/probe_lm.py', ROOT/'canonical_corpus/tokenizer_adapters.py')}
                if name == 'old':
                    previous=json.loads((path.parent/'metrics.json').read_text())['validation']['domains']
                    delta={k:result['domains'][k]['bits_per_byte']-previous[k]['bits_per_byte'] for k in result['domains']}
                    result['old_bpb_delta_from_original_metrics']=delta
                    result['old_bpb_reproduction_absolute_tolerance']=1e-5
                    assert all(abs(v)<=1e-5 for v in delta.values()), delta
                stem=f'{name}-seed{seed}-{size}'
                (a.output/(stem+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
                (a.output/(stem+'-items.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
                print(stem, {k:v['bits_per_byte'] for k,v in result['domains'].items()},flush=True)
            del model,payload;gc.collect();torch.cuda.empty_cache()
    import random
    comparison=[]
    for name in datasets:
        for seed in (1,2):
            pair=[json.loads((a.output/f'{name}-seed{seed}-{size}.json').read_text()) for size in ('32k','48k')]
            for category in ('kanbun','kakikudashi'):
                works=[{w['work_id']:w for w in result['works'] if w['category']==category} for result in pair]
                assert set(works[0])==set(works[1])
                ids=sorted(works[0]);rng=random.Random(20260909);samples=[]
                for _ in range(2000):
                    sampled=rng.choices(ids,k=len(ids))
                    bpb=[]
                    for table in works:
                        bpb.append(sum(table[i]['nll_sum'] for i in sampled)/(sum(table[i]['source_bytes'] for i in sampled)*math.log(2)))
                    samples.append(100*(bpb[1]/bpb[0]-1))
                samples.sort()
                comparison.append({'dataset':name,'seed':seed,'category':category,'works':len(ids),
                    'j32_bpb':pair[0]['domains'][category]['bits_per_byte'],
                    'j48_bpb':pair[1]['domains'][category]['bits_per_byte'],
                    'relative_percent':100*(pair[1]['domains'][category]['bits_per_byte']/pair[0]['domains'][category]['bits_per_byte']-1),
                    'paired_work_bootstrap_95_percentile_ci':[samples[49],samples[1949]],
                    'bootstrap_replicates':2000,'bootstrap_seed':20260909,
                    'ci_scope':'work sampling uncertainty conditional on one fitted model seed; not training-seed uncertainty'})
    (a.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (a.output/'done.json').write_text(json.dumps({'complete':True,'checkpoints':4,'evaluations':8})+'\n')


if __name__=='__main__':main()
