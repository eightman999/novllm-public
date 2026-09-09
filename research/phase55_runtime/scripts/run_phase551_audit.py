#!/usr/bin/env python3
"""Prepare private line-preserving hardening data and text-free audit reports."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
SOURCE_REVISION = '9f9edf0d17072d31ede9d7d9f0904967c9dbf69d'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canonical_corpus.phase551_audit import audit, rows, sha, statistics, kanbun_candidates, METHOD
from canonical_corpus.tokenizer_adapters import ReversibleSentencePieceAdapter


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def dump_rows(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))


def tokenizer_rows(path):
    adapter = object.__new__(ReversibleSentencePieceAdapter)
    with path.open() as f:
        for i, line in enumerate(f):
            yield {'id': f'J:training_sentence:{i}', 'document_id': None,
                   'text': adapter._unescape(line.removesuffix('\n'))}



def finalize_provenance(output, dataset, tokenizer_input, kanbun):
    from canonical_corpus.phase55_data import load_dataset
    train, _, dataset_manifest = load_dataset(dataset)
    report_path = output/'eval_audit.json'
    report=json.loads(report_path.read_text())
    manifest=json.loads((tokenizer_input.parent/'manifest.json').read_text())
    assert hashlib.sha256(tokenizer_input.read_bytes()).hexdigest()==manifest['input_sha256']
    plain=hashlib.sha256();framed=hashlib.sha256();chars=0;count=0
    for row in tokenizer_rows(tokenizer_input):
        payload=row['text'].encode('utf-8');plain.update(payload)
        framed.update(len(payload).to_bytes(8,'big'));framed.update(payload)
        chars+=len(row['text']);count+=1
    assert chars==manifest['source_chars']==100_000_000
    report['tokenizer_decoded_input']={'decode':'ReversibleSentencePieceAdapter._unescape; remove only physical LF training-line delimiter',
        'normalized':False,'source_chars':chars,'training_sentences':count,
        'concatenated_utf8_sha256':plain.hexdigest(),'uint64_be_byte_length_framed_utf8_sha256':framed.hexdigest(),
        'framing':'each decoded training sentence prefixed by 8-byte unsigned big-endian UTF-8 byte length',
        'input_manifest_sha256':hashlib.sha256((tokenizer_input.parent/'manifest.json').read_bytes()).hexdigest(),
        'input_hash_verified_against_recipe_manifest':True}
    report['lm_dataset_sha256']=dataset_manifest['dataset_sha256']
    for scope in report['scopes']:
        scope['train_records_with_unknown_work_identity']=sum(not r.get('document_id') for r in train) if scope['scope']=='lm_train_30m' else count
    upstream=kanbun.parents[2]/'upstream'
    source_files=[]
    for split in ('train','val','test'):
        path=upstream/(split+'.csv')
        source_files.append({'split':split,'url':f'https://raw.githubusercontent.com/nlp-waseda/Kanbun-LM/{SOURCE_REVISION}/kanbun-lm-dataset/{split}.csv',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    dump(output/'source_manifest.json',{'source':'https://github.com/nlp-waseda/Kanbun-LM','revision':SOURCE_REVISION,'files':source_files,
        'rights':'unknown; no redistribution grant located; text excluded from public artifacts',
        'preprocessing':'existing scale_special.audit_kanbun CSV import; preserve columns, split and row order; no text normalization',
        'canonical_sha256':hashlib.sha256(kanbun.read_bytes()).hexdigest()})
    dump(report_path,report)



def prepare_source(manifest_path, output):
    import urllib.request
    from canonical_corpus.scale_special import audit_kanbun
    manifest=json.loads(manifest_path.read_text())
    output.mkdir(parents=True,exist_ok=False)
    (output/'.gitignore').write_text('*\n')
    upstream=output/'upstream';upstream.mkdir()
    for item in manifest['files']:
        split=item['split']
        if split not in ('train','val','test'):
            raise ValueError('unknown source split')
        expected_prefix=f'https://raw.githubusercontent.com/nlp-waseda/Kanbun-LM/{SOURCE_REVISION}/kanbun-lm-dataset/'
        if item['url'] != expected_prefix+split+'.csv':
            raise ValueError('source URL must use audited fixed upstream commit')
        request=urllib.request.Request(item['url'],headers={'User-Agent':'NovTokenizer-Phase551-reproduce'})
        with urllib.request.urlopen(request,timeout=90) as response:
            payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=item['sha256']:
            raise ValueError('source hash mismatch')
        (upstream/(split+'.csv')).write_bytes(payload)
    audit_kanbun(upstream,output,fetch_official=False)
    canonical=output/'canonical/kanbun_lm/raw-00000.jsonl'
    if hashlib.sha256(canonical.read_bytes()).hexdigest()!=manifest['canonical_sha256']:
        raise ValueError('canonical reproduction hash mismatch')
    print(json.dumps({'canonical_sha256':manifest['canonical_sha256'],'reproduced':True}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-source',type=Path,help='fetch and prepare only the source in this text-free manifest')
    p.add_argument('--source-output',type=Path,help='new private directory for source preparation')
    p.add_argument('--dataset', type=Path, default=ROOT/'results/phase55_cuda_same_source_20260909/bundle/dataset')
    p.add_argument('--kanbun', type=Path, default=ROOT/'runs/cultural_corpus/phase46-20260907/special/kanbun_lm/canonical/kanbun_lm/raw-00000.jsonl')
    p.add_argument('--tokenizer-input', type=Path, default=ROOT/'runs/tokenizer_lab/phase5-20260907/recipes/J/train.escaped.txt')
    p.add_argument('--output', type=Path, default=ROOT/'results/phase551_eval_private')
    a = p.parse_args()
    if a.prepare_source:
        if not a.source_output: p.error('--source-output is required with --prepare-source')
        prepare_source(a.prepare_source,a.source_output);return
    a.output.mkdir(parents=True, exist_ok=False)
    # The entire destination is private; explicitly enumerated text-free report
    # files may later be copied to a reviewed public destination.
    (a.output/'.gitignore').write_text('*\n')
    canonical = list(rows(a.kanbun)); candidates = kanbun_candidates(canonical)
    source = {r['record_id']:r for r in canonical}
    old = list(rows(a.dataset/'eval.jsonl'))
    for r in old:
        if r['category'] in ('kanbun','kakikudashi'):
            rid = r['id'].split(':',2)[2]
            upstream = source[rid]
            field = 'hakubun' if r['category']=='kanbun' else 'kakikudashi'
            if r['text'] != upstream[field]:
                raise ValueError('old probe is not an exact upstream line')
            r['document_id'] = f'kanbun_lm:poetry:{upstream["poetry_id"]}'
    joined = old + candidates
    scopes = []
    all_results = []
    for name, training in [('lm_train_30m',rows(a.dataset/'train.jsonl')),
                           ('tokenizer_J_actual_sentences_100m',tokenizer_rows(a.tokenizer_input))]:
        print('audit', name, flush=True)
        result, meta = audit(joined, training, scope=name)
        dump_rows(a.output/(name+'_audit.jsonl'), result)
        all_results.extend(result); scopes.append(meta)
    flagged = {r['eval_item_id'] for r in all_results if r['status']=='match_detected'}
    excluded_works = {r['document_id'] for r in candidates if r['id'] in flagged}
    seen = set(); hardened = []
    for r in candidates:
        key = (r['category'],r['text_sha256'])
        if r['document_id'] in excluded_works or key in seen:
            continue
        seen.add(key); hardened.append(r)
    # Pairwise eval near duplicates use direct n-gram sets. No texts enter report.
    from canonical_corpus.phase551_audit import grams, normalize
    from collections import defaultdict
    index=defaultdict(set); ngs={}; duplicate_pairs=[]
    for r in candidates:
        ng=grams(normalize(r['text'])); ngs[r['id']]=ng
        others=set()
        for g in ng: others.update(index[g])
        for other in sorted(others):
            overlap=len(ng & ngs[other]); denom=min(len(ng),len(ngs[other]))
            if denom and overlap/denom >= METHOD['near_threshold']:
                duplicate_pairs.append({'eval_item_id':r['id'],'matched_eval_id':other,
                    'similarity':overlap/denom,'method':'symmetric_5gram_overlap_coefficient','threshold':.8})
        for g in ng: index[g].add(r['id'])
    dump_rows(a.output/'eval_internal_near_duplicate.jsonl',duplicate_pairs)
    old_ids={r['id'] for r in old}
    old_duplicate_pairs=[]
    for pair in duplicate_pairs:
        left=pair['eval_item_id'].replace('phase551:','phase5:',1)
        right=pair['matched_eval_id'].replace('phase551:','phase5:',1)
        if left in old_ids and right in old_ids:
            old_duplicate_pairs.append({**pair,'eval_item_id':left,'matched_eval_id':right})
    dump_rows(a.output/'old_eval_internal_near_duplicate.jsonl',old_duplicate_pairs)
    dump_rows(a.output/'hardened_eval.jsonl',hardened)
    dump_rows(a.output/'old_eval_work_joined.jsonl',old)
    manifest=[{k:v for k,v in r.items() if k!='text'} for r in hardened]
    dump_rows(a.output/'hardened_manifest.jsonl',manifest)
    inputs=[a.dataset/'train.jsonl',a.dataset/'eval.jsonl',a.kanbun,a.tokenizer_input]
    report={'method':METHOD,'old':statistics(old),'candidates':statistics(candidates),
        'hardened':statistics(hardened),'scopes':scopes,
        'excluded_work_count':len(excluded_works),'excluded_works':sorted(excluded_works),
        'flagged_eval_item_count':len(flagged),'eval_internal_near_duplicate_pairs':len(duplicate_pairs),
        'old_eval_internal_near_duplicate_pairs':len(old_duplicate_pairs),
        'selection':'exclude whole poem if any line matched LM or tokenizer input; then stable first exact text per category',
        'segmentation':'source rows retained; no concatenation; BOS per row as in Phase55',
        'source_url':'https://github.com/nlp-waseda/Kanbun-LM',
        'source_license':'unknown; inspected official repository 2026-09-09; no explicit redistribution grant located',
        'source_redistribution':'not_allowed_under_existing_registry',
        'upstream_train_is_not_novllm_train':True,
        'work_identity_limit':'poetry_id proves identity inside Kanbun-LM; cross-corpus literary identity remains unknown',
        'limitations':['5-gram containment can flag noncontiguous shared phrases; not semantic paraphrase detection',
            'Texts shorter than 5 normalized characters have unknown near/substring audit',
            'All candidates are poetry; prose and separate kundoku category are not added',
            'Source train/val/test splits preserved as metadata; this is a new exploratory eval'],
        'inputs':[{'name':x.name,'bytes':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in inputs],
        'hardened_eval_sha256':hashlib.sha256((a.output/'hardened_eval.jsonl').read_bytes()).hexdigest()}
    dump(a.output/'eval_audit.json',report)
    keys=sorted(set().union(*(r.keys() for r in all_results)))
    with (a.output/'near_duplicate_audit.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(all_results)
    finalize_provenance(a.output,a.dataset,a.tokenizer_input,a.kanbun)
    print(json.dumps({k:report[k] for k in ['excluded_work_count','flagged_eval_item_count','hardened_eval_sha256']}),flush=True)


if __name__=='__main__':
    main()
