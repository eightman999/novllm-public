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


def trajectory_plan():
    """Only already-authorized retained checkpoints; no new model training."""
    return [(size,fraction) for size in ('32k','48k') for fraction in (.2,.5,.75)] + [('64k',fraction) for fraction in (.2,.5,.75,1.)]


def checkpoint_ready(path):
    import re
    sidecar=path.with_suffix('.sha256')
    return path.is_file() and sidecar.is_file() and re.fullmatch('[0-9a-f]{64}',sidecar.read_text().strip()) is not None


def trajectory_stem(size,fraction):
    return f'hardened-seed1-{size}-fraction-{fraction:.6f}'


def score_hardened(model,tokenizer,records,cfg,device):
    prepared,checks=rt._prepare(tokenizer,records)
    items=[];accum=defaultdict(lambda:[0.,0,0,0]);work_accum=defaultdict(lambda:[0.,0,0,0])
    for row in prepared:
        measured=rt.evaluate(model,[row],cfg,device,4096)['overall']
        items.append({'eval_item_id':row['id'],'work_id':row['document_id'],'category':row['category'],**measured})
        values=[measured[key] for key in ('nll_sum','tokens','source_chars','source_bytes')]
        for bucket in (accum[row['category']],work_accum[(row['category'],row['document_id'])]):
            for i,value in enumerate(values):bucket[i]+=value
    return {'preservation':checks,'domains':{k:rt._normalized(*v) for k,v in accum.items()},
            'works':[{'category':k[0],'work_id':k[1],**rt._normalized(*v)} for k,v in work_accum.items()]},items


def trajectory_comparisons(output, final_reference, bundle=None):
    rows=[]
    for fraction in (.2,.5,.75,1.):
        available={}
        for size in ('32k','48k','64k'):
            path=output/(trajectory_stem(size,fraction)+'.json')
            if fraction==1. and size in ('32k','48k'):
                path=final_reference/f'hardened-seed1-{size}.json'
            if path.exists():
                available[size]=json.loads(path.read_text())
                if fraction==1. and size in ('32k','48k') and bundle is not None:
                    final_metrics=json.loads((bundle/'results/cuda_same_source/runs/seed1'/('j-reversible-sp-unigram-'+size)/'artifacts/metrics.json').read_text())
                    assert final_metrics['checkpoint_sha256']==available[size]['checkpoint_sha256']
                    available[size]['actual_source_chars']=final_metrics['training']['source_chars']
        for base,target in [('32k','48k'),('48k','64k')]:
            if base not in available or target not in available:continue
            left,right=available[base],available[target]
            assert left['dataset_sha256']==right['dataset_sha256']
            assert left['dataset']==right['dataset']=='hardened' and left['seed']==right['seed']==1
            for category in ('kanbun','kakikudashi'):
                a,b=left['domains'][category]['bits_per_byte'],right['domains'][category]['bits_per_byte']
                rows.append({'dataset':'hardened','seed':1,'requested_budget_fraction':fraction,'category':category,
                    'reference_candidate':base,'target_candidate':target,'reference_bpb':a,'target_bpb':b,
                    'relative_percent':100*(b/a-1),'delta_bpb':b-a,
                    'reference_actual_source_chars':left.get('actual_source_chars'),
                    'target_actual_source_chars':right.get('actual_source_chars'),
                    'reference_checkpoint_sha256':left['checkpoint_sha256'],'target_checkpoint_sha256':right['checkpoint_sha256'],
                    'work_count':len([w for w in right['works'] if w['category']==category]),
                    'intermediate_budget_note':'requested milestones matched; actual character overshoot is recorded separately'})
    rt._atomic_json(output/'comparison.json',rows)
    return rows


def run_trajectory(a):
    import fcntl
    import os
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'.controller.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('trajectory controller already owns this output')
    protocol_path=a.bundle/'protocol.json';protocol=json.loads(protocol_path.read_text())
    runtime_paths=[Path(rt.__file__),ROOT/'canonical_corpus/probe_lm.py',ROOT/'canonical_corpus/probe_lm_data.py',ROOT/'canonical_corpus/tokenizer_adapters.py']
    runtime_hashes={str(p.relative_to(ROOT)):rt._file_hash(p) for p in runtime_paths}
    for name,digest in runtime_hashes.items():
        assert digest==protocol['files'][name], 'runtime differs from fixed training protocol: '+name
    eval_path=a.eval_dir/'hardened_eval.jsonl';dataset_hash=rt._file_hash(eval_path)
    records=[json.loads(line) for line in eval_path.open(encoding='utf-8')]
    assert records and {r['category'] for r in records}=={'kanbun','kakikudashi'}
    identity={'format':'phase551-hardened-trajectory-v1','dataset_sha256':dataset_hash,
        'protocol_sha256':rt._file_hash(protocol_path),'runtime_code_sha256':runtime_hashes,
        'seed':1,'dataset':'hardened','sequence_length':4096,'plan':trajectory_plan()}
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest_path=a.output/'manifest.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())['identity_sha256']==identity_hash
    else:rt._atomic_json(manifest_path,{**identity,'identity_sha256':identity_hash})
    assert torch.cuda.is_available() and 'P100' in torch.cuda.get_device_name(0)
    rt.set_seed(20260909,threads=2)
    final_reference=a.final_reference or a.output.parent/'results'
    completed_this_invocation=0
    verified_results=set()
    while True:
        pending=[]
        for size,fraction in trajectory_plan():
            candidate='j-reversible-sp-unigram-'+size
            folder=a.bundle/'results/cuda_same_source/runs/seed1'/candidate/'artifacts'
            checkpoint=folder/f'checkpoint-fraction-{fraction:.6f}.pt'
            stem=trajectory_stem(size,fraction);result_path=a.output/(stem+'.json')
            items_path=a.output/(stem+'-items.jsonl')
            if stem in verified_results:continue
            if result_path.exists():
                previous=json.loads(result_path.read_text())
                assert previous['trajectory_identity_sha256']==identity_hash
                assert rt._file_hash(items_path)==previous['items_sha256']
                assert rt._file_hash(checkpoint)==previous['checkpoint_sha256']
                verified_results.add(stem)
                continue
            if not checkpoint_ready(checkpoint):
                pending.append(stem);continue
            started=time.monotonic();digest=rt._file_hash(checkpoint)
            assert digest==checkpoint.with_suffix('.sha256').read_text().strip()
            saved_config=json.loads((folder/'config.json').read_text())
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            cfg=LMConfig(**payload['model_config'])
            assert cfg.to_dict()==protocol['tokenizers'][candidate]['model_config']
            assert payload['identity']==saved_config['identity']
            for path in runtime_paths:
                assert rt._file_hash(path)==saved_config['code_sha256'][path.name]
            assert saved_config['dataset_sha256']==protocol['dataset_sha256']
            actual_chars=payload['progress']['source_chars'];budget=payload['scheduler']['source_char_budget']
            assert budget==30_000_000 and actual_chars>=budget*fraction
            model=ProbeLM(cfg).cuda();model.load_state_dict(payload['model']);model.eval()
            tokenizer_path=a.bundle/protocol['tokenizers'][candidate]['path']
            assert rt._file_hash(tokenizer_path)==protocol['tokenizers'][candidate]['sha256']==saved_config['tokenizer_sha256']
            tokenizer=ReversibleSentencePieceAdapter(tokenizer_path)
            measured,items=score_hardened(model,tokenizer,records,cfg,torch.device('cuda:0'))
            assert rt._file_hash(checkpoint)==digest
            assert rt._file_hash(eval_path)==dataset_hash and rt._file_hash(protocol_path)==identity['protocol_sha256']
            for path in runtime_paths:assert rt._file_hash(path)==runtime_hashes[str(path.relative_to(ROOT))]
            temp=items_path.with_suffix('.jsonl.tmp')
            temp.write_text(''.join(json.dumps(item,ensure_ascii=False)+'\n' for item in items))
            temp.replace(items_path)
            result={**measured,'candidate':candidate,'seed':1,'dataset':'hardened','dataset_sha256':dataset_hash,
                'trajectory_identity_sha256':identity_hash,'requested_budget_fraction':fraction,
                'actual_budget_fraction':actual_chars/budget,'actual_source_chars':actual_chars,
                'train_tokens':payload['progress']['tokens'],'train_steps':payload['progress']['steps'],
                'source_char_budget':budget,'checkpoint_sha256':digest,'checkpoint_unchanged_after_evaluation':True,
                'runtime_code_sha256':runtime_hashes,'items_sha256':rt._file_hash(items_path),
                'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
                'evaluation_seconds':time.monotonic()-started,'segmentation':'original lines; unchanged Phase55 evaluate(); BOS per line'}
            rt._atomic_json(result_path,result)
            print(stem,{k:v['bits_per_byte'] for k,v in result['domains'].items()},flush=True)
            del model,payload;gc.collect();torch.cuda.empty_cache()
            completed_this_invocation+=1
            verified_results.add(stem)
            trajectory_comparisons(a.output,final_reference,a.bundle)
            if a.max_checkpoints and completed_this_invocation>=a.max_checkpoints:
                return
        remaining=[trajectory_stem(size,fraction) for size,fraction in trajectory_plan() if not (a.output/(trajectory_stem(size,fraction)+'.json')).exists()]
        rt._atomic_json(a.output/'state.json',{'complete':not remaining,'pending':remaining,'controller_pid':os.getpid(),'updated_unix':time.time()})
        trajectory_comparisons(a.output,final_reference,a.bundle)
        if not remaining:
            rt._atomic_json(a.output/'done.json',{'complete':True,'evaluations':len(trajectory_plan()),'identity_sha256':identity_hash});return
        if a.once:return
        time.sleep(30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--eval-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--trajectory',action='store_true',help='evaluate retained seed1 milestones on hardened data; wait for J64 checkpoints')
    ap.add_argument('--once',action='store_true',help='trajectory: process available checkpoints and return without waiting')
    ap.add_argument('--max-checkpoints',type=int,help='trajectory: stop after this many new evaluations (smoke/restart)')
    ap.add_argument('--final-reference',type=Path,help='previous unchanged final hardened result directory')
    a=ap.parse_args()
    if a.trajectory:return run_trajectory(a)
    a.output.mkdir(parents=True,exist_ok=False)
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
