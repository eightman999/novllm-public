#!/usr/bin/env python3
"""Publish contamination measurements without corpus text or private identifiers."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

CATEGORIES=set('aa aozora_kyu_kyu aozora_shin_kyu aozora_shin_shin gyaru historical_kana kakikudashi kanbun old_orthography technical unicode_edge web_novel whitespace'.split())
SCOPES={'lm_train_30m','tokenizer_J_actual_sentences_100m'}
FLAGS=('train_exact_match','train_normalized_exact_match','train_long_substring_match','train_short_substring_match','train_near_match')
IDS=('eval_item_id','matched_train_id','exact_matched_train_id','normalized_matched_train_id','long_substring_matched_train_id','substring_matched_train_id')
SOURCE='https://github.com/nlp-waseda/Kanbun-LM'

def opaque(value):
    if value is None or value=='unknown':return value
    if not isinstance(value,str):raise ValueError('string identity required')
    return 'sha256:'+hashlib.sha256(value.encode()).hexdigest()

def digest(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('SHA256 required')
    return value

def numeric(value):
    if value is not None and (not isinstance(value,(float,int)) or not math.isfinite(value)):raise ValueError('finite number required')
    return value

def records(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():yield json.loads(line)

def emit(folder,name,rows):
    if len(rows) <= 1000:
        (folder/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    if rows:
        with (folder/(name+'.csv')).open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def export(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    audit_rows=[]
    for scope in sorted(SCOPES):
        for r in records(source/(scope+'_audit.jsonl')):
            if r['category'] not in CATEGORIES or r['comparison_scope']!=scope:raise ValueError('unexpected scope/category')
            row={k:opaque(r.get(k)) for k in IDS}
            row.update(source_work_identity=opaque(r.get('source_work_identity')),category=r['category'],comparison_scope=scope)
            for key in FLAGS:
                value=r.get(key)
                if value is not None and type(value) is not bool:raise ValueError('boolean match field required')
                row[key]=value
            row['max_similarity']=numeric(r['max_similarity'])
            if row['max_similarity'] is not None and not 0<=row['max_similarity']<=1:raise ValueError('similarity range')
            if r['threshold']!=.8:raise ValueError('threshold changed')
            row.update(method='directional unique character 5-gram containment in one train record',threshold=.8)
            if r['status'] not in {'match_detected','no_detected_text_match_work_unknown','insufficient_length_work_unknown'}:raise ValueError('unknown audit status')
            if r['work_overlap'] is not True and r['work_overlap']!='unknown':raise ValueError('unverified work disjointness')
            row.update(status=r['status'],work_overlap=r['work_overlap']);audit_rows.append(row)
    emit(output,'near_duplicate_audit',audit_rows)
    report=json.loads((source/'eval_audit.json').read_text())
    stats=[]
    for partition in ('old','candidates','hardened'):
        for category,r in sorted(report[partition].items()):
            if category not in CATEGORIES:raise ValueError('unexpected category')
            stats.append({'partition':partition,'category':category,**{k:numeric(r.get(k)) for k in ('documents','characters','bytes','known_works','unknown_work_documents','max_work_character_fraction','exact_duplicate_extra_records')}})
    emit(output,'eval_audit',stats)
    manifest=[]
    for r in records(source/'hardened_manifest.jsonl'):
        if r['category'] not in {'kanbun','kakikudashi'} or r['upstream_split'] not in {'train','val','test'}:raise ValueError('unexpected source')
        public_record=r['source_record_id']
        if not re.fullmatch(r'kanbun_lm:(train|val|test):[0-9]+',public_record):raise ValueError('public source record identity invalid')
        manifest.append({'eval_item_id':opaque(r['id']),'source_work_identity':opaque(r['document_id']),
            'category':r['category'],'source_url':SOURCE,'source_record_id':public_record,'upstream_split':r['upstream_split'],
            'text_sha256':digest(r['text_sha256']),'source_chars':numeric(r['source_chars']),'source_bytes':numeric(r['source_bytes'])})
    emit(output,'hardened_manifest',manifest)
    pairs=[]
    for r in records(source/'eval_internal_near_duplicate.jsonl'):
        if r['threshold']!=.8:raise ValueError('threshold changed')
        pairs.append({'eval_item_id':opaque(r['eval_item_id']),'matched_eval_id':opaque(r['matched_eval_id']),
            'similarity':numeric(r['similarity']),'method':'symmetric_5gram_overlap_coefficient','threshold':.8})
    emit(output,'eval_internal_near_duplicate',pairs)
    old_pairs=[]
    old_path=source/'old_eval_internal_near_duplicate.jsonl'
    if old_path.exists():
        for r in records(old_path):
            if r['threshold']!=.8:raise ValueError('threshold changed')
            old_pairs.append({'eval_item_id':opaque(r['eval_item_id']),'matched_eval_id':opaque(r['matched_eval_id']),'similarity':numeric(r['similarity']),'method':'symmetric_5gram_overlap_coefficient','threshold':.8})
        emit(output,'old_eval_internal_near_duplicate',old_pairs)
    metadata={'id_transform':'sha256 of original UTF-8 identifier; unknown stays unknown; no source text',
        'method':{'audit_normalization':'NFKC then remove Unicode whitespace; evaluation text unchanged','ngram_n':5,'near_threshold':.8,'long_substring_chars':50,'short_substring_min_chars':5},
        'selection':'exclude whole poem on any detected LM/tokenizer text match; stable first exact text per category',
        'excluded_work_count':numeric(report['excluded_work_count']), 'flagged_eval_item_count':numeric(report['flagged_eval_item_count']),
        'eval_internal_near_duplicate_pairs':numeric(report['eval_internal_near_duplicate_pairs']),
        'hardened_eval_sha256':digest(report['hardened_eval_sha256']),'lm_dataset_sha256':digest(report['lm_dataset_sha256']),
        'scopes':[{k: (r[k] if k=='scope' and r[k] in SCOPES else numeric(r[k])) for k in ('scope','train_records','train_chars','train_records_with_unknown_work_identity')} for r in report['scopes']],
        'limitations':['cross-corpus work identity remains unknown even without a text match','shorter than 5 normalized chars have unknown near/substring audit','poetry only; no new prose style or separate kundoku category','near duplicates inside evaluation are reported, not all removed; bootstrap by poem accounts for within-poem clustering only','raw text not redistributed; sources have no established redistribution grant'],
        'input_hashes':[{'sha256':digest(r['sha256']),'bytes':numeric(r['bytes'])} for r in report['inputs']]}
    (output/'audit_method.json').write_text(json.dumps(metadata,indent=2)+'\n')
    eval_dir=source/'checkpoint_evaluation'
    if (eval_dir/'done.json').exists():
        summaries=[];work_scores=[];checks=[]
        for dataset in ('old','hardened'):
            for seed in (1,2):
                for size in ('32k','48k'):
                    r=json.loads((eval_dir/f'{dataset}-seed{seed}-{size}.json').read_text())
                    identity={'dataset':dataset,'seed':seed,'candidate':'j-reversible-sp-unigram-'+size}
                    assert all(r[k]==v for k,v in identity.items())
                    assert r['checkpoint_unchanged_after_evaluation'] is True
                    assert r['preservation']['exact_preservation']==1 and r['preservation']['unknown_tokens']==0
                    for domain,score in r['domains'].items():
                        if domain not in {'kanbun','kakikudashi'}:raise ValueError('unexpected hardening domain')
                        summaries.append({**identity,'category':domain,**{k:numeric(score[k]) for k in ('bits_per_byte','nll_sum','tokens','source_chars','source_bytes')}})
                    work_order={domain:{work:i for i,work in enumerate(sorted(w['work_id'] for w in r['works'] if w['category']==domain))} for domain in ('kanbun','kakikudashi')}
                    for w in r['works']:
                        if w['category'] not in {'kanbun','kakikudashi'}:raise ValueError('unexpected work domain')
                        work_scores.append({**identity,'category':w['category'],'source_work_identity':opaque(w['work_id']),'bootstrap_work_index':work_order[w['category']][w['work_id']],**{k:numeric(w[k]) for k in ('bits_per_byte','nll_sum','tokens','source_chars','source_bytes')}})
                    checks.append({**identity,'checkpoint_sha256':digest(r['checkpoint_sha256']),'dataset_sha256':digest(r['dataset_sha256']),
                        'checkpoint_unchanged':True,'exact_preservation':1.0,'unknown_tokens':0,'evaluation_seconds':numeric(r['evaluation_seconds']),
                        'evaluation_gpu':'Tesla P100-PCIE-16GB',
                        **{'code_'+Path(k).name:digest(v) for k,v in r['runtime_code_sha256'].items() if k in ('canonical_corpus/phase55_same_source_runtime.py','canonical_corpus/probe_lm.py','canonical_corpus/tokenizer_adapters.py')}})
        comparisons=[]
        for r in json.loads((eval_dir/'comparison.json').read_text()):
            if r['dataset'] not in {'old','hardened'} or r['seed'] not in (1,2) or r['category'] not in {'kanbun','kakikudashi'}:raise ValueError('unexpected comparison identity')
            lo,hi=r['paired_work_bootstrap_95_percentile_ci']
            comparisons.append({**{k:r[k] for k in ('dataset','seed','category')},**{k:numeric(r[k]) for k in ('works','j32_bpb','j48_bpb','relative_percent','bootstrap_replicates','bootstrap_seed')},'ci_lower_percent':numeric(lo),'ci_upper_percent':numeric(hi)})
        for name,table in [('evaluation_summary',summaries),('work_scores',work_scores),('evaluation_checks',checks),('hardened_comparison',comparisons)]:emit(output,name,table)
    return {'audit_rows':len(audit_rows),'manifest_rows':len(manifest),'stats_rows':len(stats),'near_pairs':len(pairs)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(export(a.input,a.output)))
