#!/usr/bin/env python3
"""Isolated fresh CUDA same-source pairs. No cross-backend checkpoint loading."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from canonical_corpus import phase55_same_source_runtime as rt
from canonical_corpus.phase55_data import load_dataset
from canonical_corpus.probe_lm import LMConfig, ProbeLM
RUN_ROOT=ROOT
OUT=RUN_ROOT/'results/cuda_same_source'
def dump(p,v): rt._atomic_json(p,v)
def protocol(): return json.loads((ROOT/'protocol.json').read_text())
def run(candidate,seed,smoke=False):
    p=protocol(); info=p['tokenizers'][candidate]
    # Preserve exact input hashes; only the audited runtime closure is distributed.
    for name,h in p['files'].items():
        if name.startswith(('dataset/', 'tokenizers/')):
            assert rt._file_hash(RUN_ROOT/name)==h,name
    provenance=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for entry in provenance['files']:
        assert rt._file_hash(ROOT/entry['path'])==entry['published_sha256'],entry['path']
    train,evaluation,m=load_dataset(RUN_ROOT/'dataset')
    assert m['dataset_sha256']==p['dataset_sha256']
    assert sum(len(r['text']) for r in train)==30_000_000
    assert hashlib.sha256(json.dumps([(r['id'],r['text_sha256']) for r in train],ensure_ascii=False).encode()).hexdigest()==p['train_order_sha256']
    cfg=LMConfig(**info['model_config'])
    controls={**p['controls'],'seed':seed,'device':'cuda:0','resume':False,'verify_resume':True,
              'curve_fractions':[.19,.2,.5,.75,1.0],'retain_curve_checkpoints':True}
    folder=OUT/('smoke-deterministic' if smoke else 'runs')/f'seed{seed}'/candidate
    folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'hardware.json',{'gpu':torch.cuda.get_device_name(0),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
         'torch':str(torch.__version__),'fresh':True,'seed':seed,'dtype':'float32','started_unix':time.time()})
    if smoke:
        # Real full-context memory/optimizer smoke before the shorter runtime E2E.
        rt.set_seed(seed,threads=2)
        model=ProbeLM(cfg).cuda(); opt=rt._optimizer(model,controls)
        x=torch.ones((1,4096),dtype=torch.long,device='cuda'); batch=(x,x,torch.ones_like(x,dtype=torch.bool))
        controls['_scheduler_position']=4096
        loss=rt._step(model,opt,batch,controls,0)
        dump(folder/'full_context.json',{'loss':loss,'peak_vram':torch.cuda.max_memory_allocated(),'context':4096})
        del model,opt,x,batch
        import gc
        gc.collect();torch.cuda.empty_cache()
        train=rt._source_prefix(train,8192); evaluation=evaluation[:1]
        controls.update(stage='smoke',run_kind='smoke',max_steps=100,source_char_budget=8192,warmup_source_chars=82,curve_fractions=[1.0])
    metrics=rt.run_probe(candidate=candidate,tokenizer_path=RUN_ROOT/info['path'],train_records=train,eval_records=evaluation,
        model_config=cfg,run_config=controls,output_dir=folder/'artifacts',dataset_hash=m['dataset_sha256'])
    assert metrics['complete'] and not metrics['resumed_from_checkpoint']
    assert metrics['checks']['optimizer_resume_continuation']
    if not smoke: assert metrics['main_comparison_eligible'] and metrics['training']['source_chars']==30_000_000
    dump(folder/'done.json',{'complete':True,'finished_unix':time.time()})
def report(seed):
    rows=[]
    for k in ('32k','48k'):
        c='j-reversible-sp-unigram-'+k
        folder=OUT/'runs'/f'seed{seed}'/c
        if not (folder/'done.json').exists():return
        m=json.loads((folder/'artifacts/metrics.json').read_text());t=m['training']; v=m['validation']
        assert m['main_comparison_eligible']
        rows.append({'candidate':c,'seed':seed,'gpu':json.loads((folder/'hardware.json').read_text())['gpu'],
        'overall_bpb':v['overall']['bits_per_byte'],'domains':v['domains'],'source_chars':t['source_chars'],
        'training_tokens':t['tokens'],'chars_per_token':t['source_chars']/t['tokens'],'wall_clock':m['system']['invocation_wall_seconds'],
        'train_wall_clock':t['train_wall_seconds'],'tokens_per_sec':t['tokens_per_second'],'chars_per_sec':t['chars_per_second'],
        'peak_vram':m['system']['peak_vram_bytes'],'final_train_loss':t['last_train_loss'],'parameters':m['parameter_report']})
    a,b=rows
    delta={key:b[key]-a[key] for key in ('overall_bpb','training_tokens','wall_clock','train_wall_clock')}
    delta['relative_bpb_percent']=100*(b['overall_bpb']/a['overall_bpb']-1)
    delta['domains']={d:{'delta_bpb':b['domains'][d]['bits_per_byte']-a['domains'][d]['bits_per_byte'],
        'relative_percent':100*(b['domains'][d]['bits_per_byte']/a['domains'][d]['bits_per_byte']-1)} for d in a['domains']}
    dump(OUT/f'comparison-seed{seed}.json',{'runs':rows,'j48_minus_j32':delta,'freeze_ready':False,'phase6_started':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['smoke','pair','run','report']);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--candidate',default='j-reversible-sp-unigram-48k');ap.add_argument('--run-root',type=Path,default=ROOT);a=ap.parse_args()
    RUN_ROOT=a.run_root.resolve(); OUT=RUN_ROOT/'results/cuda_same_source'
    OUT.mkdir(parents=True,exist_ok=True)
    if a.command=='report':report(a.seed)
    elif a.command in ('smoke','run'):run(a.candidate,a.seed,a.command=='smoke')
    else:
        for k in ('32k','48k'):
            c='j-reversible-sp-unigram-'+k
            with (OUT/f'seed{a.seed}-{k}.log').open('x') as log:
                subprocess.run([sys.executable,__file__,'run','--seed',str(a.seed),'--candidate',c,'--run-root',str(RUN_ROOT)],stdout=log,stderr=subprocess.STDOUT,check=True)
        report(a.seed)
