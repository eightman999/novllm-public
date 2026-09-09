#!/usr/bin/env python3
"""Derive the preregistered one-seed J64 decision from public measurement tables."""
import argparse
import csv
import json
import math
import re
from pathlib import Path


def classify(relative_bpb_percent):
    """User's fixed thresholds; no automatic follow-up experiments."""
    if not math.isfinite(relative_bpb_percent):
        raise ValueError('finite BPB delta required')
    if relative_bpb_percent <= -1.0:
        return 'A: J64 second seed推奨'
    if relative_bpb_percent <= -0.5:
        return 'B: J64 second seed検討'
    if relative_bpb_percent < 0.5:
        return 'C: J48優先'
    return 'D: J64打ち切り'


def change(value, baseline):
    if value is None or baseline is None:
        return None
    if baseline <= 0:
        raise ValueError('positive comparison denominator required')
    return 100 * (value / baseline - 1)


def write_table(folder, name, rows):
    (folder / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if rows:
        with (folder / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


REQUIRED_DOMAINS = set('aa aozora_kyu_kyu aozora_shin_kyu aozora_shin_shin gyaru historical_kana kakikudashi kanbun old_orthography technical unicode_edge web_novel whitespace'.split())
REQUIRED_FRACTIONS = {0.19, 0.20, 0.50, 0.75, 1.0}


def required_equal(rows, baseline, keys, label):
    for key in keys:
        if rows.get(key) is None or rows.get(key) != baseline.get(key):
            raise ValueError(f'{label} mismatch or missing: {key}')


def sha256(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def analyze(folder, seed=1):
    folder = Path(folder)
    names = [f'j-reversible-sp-unigram-{n}k' for n in (32, 48, 64)]
    tables = {k: json.loads((folder / f'{k}.json').read_text()) for k in ('summary', 'parameters', 'hardware', 'conditions', 'domain_bpb', 'checkpoints', 'provenance')}
    def cohort(table):
        selected = [r for r in tables[table] if r['seed'] == seed and r['candidate'] in names]
        if len(selected) != 3 or {r['candidate'] for r in selected} != set(names):
            raise ValueError(f'one complete J32/J48/J64 cohort required in {table}')
        return {r['candidate']: r for r in selected}
    summary, params, hardware, controls, provenance = [cohort(t) for t in ('summary', 'parameters', 'hardware', 'conditions', 'provenance')]
    tokenizer_rows = json.loads((folder / 'tokenizer_config.json').read_text())
    tokenizer_rows = [r for r in tokenizer_rows if r.get('candidate') in names]
    if len(tokenizer_rows) != 3 or {r['candidate'] for r in tokenizer_rows} != set(names):
        raise ValueError('one tokenizer config per candidate required')
    tokenizers = {r['candidate']: r for r in tokenizer_rows}
    reference_tokenizer = tokenizers[names[1]]
    for size, name in zip((32000, 48000, 64000), names):
        tok = tokenizers[name]
        if params[name].get('vocab_size') != size or tok.get('actual_vocab_size') != size or tok.get('trainer_args', {}).get('vocab_size') != size:
            raise ValueError('candidate vocabulary mismatch')
        if not sha256(tok.get('tokenizer_sha256')) or tok['tokenizer_sha256'] != provenance[name].get('tokenizer_sha256'):
            raise ValueError('tokenizer artifact mismatch')
        for key in ('input_sha256', 'corpus_manifest_sha256'):
            if not sha256(tok.get(key)):
                raise ValueError(f'tokenizer input missing: {key}')
        required_equal(tok, reference_tokenizer, ('input_sha256', 'corpus_manifest_sha256', 'source_chars', 'seed', 'adapter', 'adapter_version', 'recipe', 'library_version'), 'tokenizer input')
        args = tok.get('trainer_args', {})
        reference_args = reference_tokenizer.get('trainer_args', {})
        if args.get('normalization_rule_name') != 'identity' or tok.get('recipe') != 'J' or tok.get('adapter_version') != 'escape-e000-v2':
            raise ValueError('tokenizer normalization recipe mismatch')
        if {k:v for k,v in args.items() if k != 'vocab_size'} != {k:v for k,v in reference_args.items() if k != 'vocab_size'}:
            raise ValueError('tokenizer preprocessing mismatch')
        for table, key, expected in (('domain_bpb', 'domain', REQUIRED_DOMAINS), ('checkpoints', 'requested_budget_fraction', REQUIRED_FRACTIONS)):
            rows = [r for r in tables[table] if r.get('candidate') == name and r.get('seed') == seed]
            if len(rows) != len(expected) or {r.get(key) for r in rows} != expected:
                raise ValueError(f'missing or duplicate {table} cohort')
    baseline = summary[names[1]]
    for name in names:
        r = summary[name]
        if r['complete'] is not True or r['train_source_chars'] != 30_000_000:
            raise ValueError('incomplete source budget')
        for key in ('dataset_sha256', 'records_sha256', 'code_phase55_same_source_runtime.py', 'code_probe_lm.py', 'code_probe_lm_data.py', 'code_tokenizer_adapters.py'):
            if provenance[name].get(key) is None or provenance[name].get(key) != provenance[names[1]].get(key):
                raise ValueError(f'lineage mismatch: {key}')
        for key in ('train_source_chars', 'train_source_bytes', 'source_chars', 'source_bytes'):
            if r[key] != baseline[key]:
                raise ValueError(f'comparison denominator mismatch: {key}')
        if hardware[name].get('dtype') != 'float32':
            raise ValueError('float32 comparison required')
        if hardware[name]['gpu'] is None or hardware[name]['gpu'] != hardware[names[1]]['gpu']:
            raise ValueError('same GPU comparison required')
        if abs(params[name]['total'] - 150_000_000) / 150_000_000 > 0.001:
            raise ValueError('parameter budget mismatch')
        required_equal(params[name], params[names[1]], ('hidden_size','num_layers','num_heads','num_kv_heads','context_length','bos_id','eos_id','pad_id','tie_embeddings','activation','norm','positional_encoding'), 'architecture')
        required_equal(controls[name], controls[names[1]], ('learning_rate','weight_decay','gradient_clip','sequence_length','batch_size','source_char_budget','warmup_source_chars','deterministic_algorithms','total_steps','warmup_steps','scheduler_axis','regime','is_smoke','verify_resume'), 'control')
        if controls[name]['scheduler_axis'] != 'source_chars' or controls[name]['regime'] != 'same_source_characters' or controls[name]['is_smoke'] is not False:
            raise ValueError('non-comparable training regime')
    delta = change(summary[names[2]]['bits_per_byte'], baseline['bits_per_byte'])
    decision = {'seed': seed, 'j64_vs_j48_bpb_percent': delta, 'decision': classify(delta),
                'thresholds_fixed_before_result': {'A_lte':-1.0,'B_lte':-0.5,'C_lt':0.5,'D_gte':0.5},
                'scope':'one exploratory seed; total parameters matched by changing FFN width',
                'second_seed_started':False,'freeze_ready':False,'phase6_started':False}
    comparison = []
    for name in names:
        r, p = summary[name], params[name]
        comparison.append({'candidate':name,'seed':seed,'overall_bpb':r['bits_per_byte'],
            'bpb_relative_to_j32_percent':change(r['bits_per_byte'],summary[names[0]]['bits_per_byte']),
            'bpb_relative_to_j48_percent':change(r['bits_per_byte'],baseline['bits_per_byte']),
            'train_tokens':r['train_tokens'],'train_chars_per_token':r['train_source_chars']/r['train_tokens'],
            'train_bytes_per_token':r['train_source_bytes']/r['train_tokens'],
            'train_seconds':r['train_train_wall_seconds'],'invocation_seconds':r['invocation_wall_seconds'],
            'train_time_relative_to_j48_percent':change(r['train_train_wall_seconds'],baseline['train_train_wall_seconds']),
            'peak_vram_bytes':r['peak_vram_bytes'],'total_params':p['total'],'embedding_params':p['embedding'],
            'non_embedding_params':p['non_embedding'],'embedding_ratio':p['embedding_ratio'],
            'derived_flops':r['train_derived_flops'],'measured_flops':r['train_measured_flops']})
    domains=[]
    for row in tables['domain_bpb']:
        if row['seed'] != seed or row['candidate'] != names[2]:continue
        old={n:next(x for x in tables['domain_bpb'] if x['candidate']==n and x['seed']==seed and x['domain']==row['domain']) for n in names[:2]}
        if any((x['source_bytes'],x['source_chars']) != (row['source_bytes'],row['source_chars']) for x in old.values()):raise ValueError('domain denominator mismatch')
        domains.append({'domain':row['domain'],'source_chars':row['source_chars'],'source_bytes':row['source_bytes'],
            'j32_bpb':old[names[0]]['bits_per_byte'],'j48_bpb':old[names[1]]['bits_per_byte'],'j64_bpb':row['bits_per_byte'],
            'j64_vs_j32_percent':change(row['bits_per_byte'],old[names[0]]['bits_per_byte']),
            'j64_vs_j48_percent':change(row['bits_per_byte'],old[names[1]]['bits_per_byte'])})
    curves=[]
    for row in tables['checkpoints']:
        if row['seed']!=seed or row['candidate']!=names[2]:continue
        old=next(x for x in tables['checkpoints'] if x['candidate']==names[1] and x['seed']==seed and x['requested_budget_fraction']==row['requested_budget_fraction'])
        curves.append({'requested_fraction':row['requested_budget_fraction'],'j48_actual_chars':old['source_chars'],'j64_actual_chars':row['source_chars'],
            'j48_bpb':old['eval_bits_per_byte'],'j64_bpb':row['eval_bits_per_byte'],
            'j64_vs_j48_percent':change(row['eval_bits_per_byte'],old['eval_bits_per_byte'])})
    for name,rows in [('j64_comparison',comparison),('j64_domain_delta',domains),('j64_checkpoint_delta',curves)]:write_table(folder,name,rows)
    (folder/'decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    lines=['# Phase 5.5.1 J64 exploratory result','','## Measured','',f"Same source / approximately 150M total parameters; seed {seed}, {hardware[names[1]]['gpu']}.",'', '| Metric | J32 | J48 | J64 |','|---|---:|---:|---:|']
    for key in ('overall_bpb','train_tokens','train_chars_per_token','train_bytes_per_token','train_seconds','invocation_seconds','peak_vram_bytes','total_params','embedding_params','non_embedding_params','embedding_ratio'):
        values=[r[key] for r in comparison]
        lines.append('| '+key+' | '+' | '.join('unknown' if x is None else str(round(x,6)) if isinstance(x,float) else str(x) for x in values)+' |')
    lines+=['','## Derived','',f"J64 vs J48 BPB: {delta:+.4f}%. Decision: **{decision['decision']}**.",'','All percentages are recomputed from measured fields. FLOPs are analytic estimates, not hardware compute measurements.','','## Interpretation','','Apply the fixed threshold to overall BPB. Domain regressions, embedding ratio and timing remain separate evidence; no second seed or freeze is automatically authorized. This is a model/tokenizer system comparison: FFN width changes to keep total parameters fixed.','','## Not yet verified','','One seed cannot establish seed robustness. Original domain evaluation remains distinct from the hardened corpus. Checkpoint percentages are first boundary crossings, with actual character counts preserved; they are not exactly equal character positions. No 300M run, tokenizer freeze or Phase 6 was performed.']
    (folder/'J64_REPORT.md').write_text('\n'.join(lines)+'\n')
    return decision

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    print(json.dumps(analyze(args.results,args.seed),ensure_ascii=False))
