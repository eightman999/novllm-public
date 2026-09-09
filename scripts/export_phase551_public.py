#!/usr/bin/env python3
"""Export aggregate measurements only; never copy raw input objects or text."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile

CANDIDATES = {f'j-reversible-sp-unigram-{n}k' for n in (32, 48, 64)}
DOMAINS = set('aa aozora_kyu_kyu aozora_shin_kyu aozora_shin_shin gyaru historical_kana kakikudashi kanbun old_orthography technical unicode_edge web_novel whitespace'.split())
NUMERIC = 'bits_per_byte nll_sum source_bytes source_chars tokens token_nll'.split()
MODEL = 'vocab_size hidden_size num_layers num_heads num_kv_heads ffn_size context_length bos_id eos_id pad_id tie_embeddings'.split()
CONTROLS = 'batch_size decode_steps deterministic_algorithms gradient_clip is_smoke learning_rate retain_curve_checkpoints seed sequence_length source_char_budget threads total_steps verify_resume warmup_source_chars warmup_steps weight_decay'.split()

def numbers(obj, keys):
    out = {}
    for key in keys:
        v = obj.get(key)
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise ValueError(f'non-numeric allowlisted field: {key}')
        out[key] = v
    return out

def enums(obj, allowed):
    result = {}
    for key, values in allowed.items():
        value = obj.get(key)
        if value is not None and value not in values:
            raise ValueError(f'unknown enum: {key}')
        result[key] = value
    return result


def export_tokenizers(source, output):
    rows = []
    for candidate in sorted(CANDIDATES):
        path = Path(source) / candidate / 'config.json'
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        trainer = config.get('trainer_args', {})
        rows.append({'candidate': candidate, **numbers(config, ['seed', 'source_chars', 'actual_vocab_size']),
            **enums(config, {'adapter': {'reversible'}, 'adapter_version': {'escape-e000-v2'}, 'recipe': {'J'}}),
            'input_sha256': digest(config.get('input_sha256')), 'corpus_manifest_sha256': digest(config.get('corpus_manifest_sha256')),
            'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'tokenizer_sha256': digest(config.get('artifacts_sha256', {}).get('tokenizer.model')),
            'library_version': version(config.get('library_version')),
            'trainer_args': {**numbers(trainer, 'add_dummy_prefix allow_whitespace_only_pieces bos_id byte_fallback character_coverage eos_id hard_vocab_limit input_sentence_size max_sentence_length num_threads pad_id remove_extra_whitespaces seed_sentencepiece_size shrinking_factor shuffle_input_sentence split_by_unicode_script split_by_whitespace unk_id vocab_size'.split()),
                **enums(trainer, {'model_type': {'unigram'}, 'normalization_rule_name': {'identity'}})}})
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'tokenizer_config.json').write_text(json.dumps(rows, indent=2) + '\n')


def digest(value):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('invalid SHA256')
    return value

def version(value):
    return value if isinstance(value, str) and re.fullmatch(r'[0-9][0-9a-zA-Z.+_-]{0,60}', value) else None

def ratio(a, b):
    return a / b if a is not None and b else None

def validate_score(score):
    score = numbers(score, NUMERIC)
    if score['nll_sum'] is not None and score['source_bytes']:
        bpb = score['nll_sum'] / score['source_bytes'] / math.log(2)
        if not math.isclose(bpb, score['bits_per_byte'], rel_tol=1e-10):
            raise ValueError('BPB is inconsistent with NLL / bytes')
    return score

def aggregate(runs):
    tables = {k: [] for k in ('summary', 'domain_bpb', 'checkpoints', 'hardware', 'parameters', 'compression', 'conditions', 'provenance', 'checkpoint_domains')}
    seen = set()
    for metrics, config, hardware, source_hash in runs:
        candidate = metrics['candidate']
        seed = metrics['seed']
        if candidate not in CANDIDATES or type(seed) is not int or seed < 0:
            raise ValueError('invalid run identity')
        if (candidate, seed) in seen:
            raise ValueError('duplicate run identity')
        seen.add((candidate, seed))
        identity = {'candidate': candidate, 'seed': seed}
        training = metrics.get('training', {})
        validation = metrics.get('validation', {})
        overall = validate_score(validation.get('overall', {}))
        train = numbers(training, 'source_chars source_bytes tokens steps train_wall_seconds tokens_per_second chars_per_second derived_flops measured_flops'.split())
        tables['summary'].append({**identity, 'complete': metrics.get('complete') is True, **overall, **{'train_' + k: v for k, v in train.items()}, **numbers(metrics.get('system', {}), ['peak_vram_bytes', 'invocation_wall_seconds'])})
        for domain, values in sorted(validation.get('domains', {}).items()):
            if domain not in DOMAINS:
                raise ValueError('unknown domain; review before publishing')
            tables['domain_bpb'].append({**identity, 'domain': domain, **validate_score(values)})
        for curve in metrics.get('learning_curves', []):
            tables['checkpoints'].append({**identity, **numbers(curve, 'requested_budget_fraction actual_budget_fraction source_chars source_bytes tokens steps training_hours derived_flops'.split()), **{'eval_' + k: v for k, v in validate_score(curve.get('validation', {}).get('overall', {})).items()}})
        for curve in metrics.get('learning_curves', []):
            for domain, values in sorted(curve.get('validation', {}).get('domains', {}).items()):
                if domain not in DOMAINS:
                    raise ValueError('unknown curve domain')
                tables['checkpoint_domains'].append({**identity, 'domain': domain, **numbers(curve, ['requested_budget_fraction', 'actual_budget_fraction']), 'train_source_chars': numbers(curve, ['source_chars'])['source_chars'], **validate_score(values)})
        params = numbers(metrics.get('parameter_report', {}), ['total', 'embedding', 'non_embedding', 'lm_head'])
        if all(params[k] is not None for k in ('total', 'embedding', 'non_embedding')) and params['total'] != params['embedding'] + params['non_embedding']:
            raise ValueError('parameter breakdown mismatch')
        tables['parameters'].append({**identity, **params, 'embedding_ratio': ratio(params['embedding'], params['total']), **numbers(metrics.get('model_config', {}), MODEL), **enums(metrics.get('model_config', {}), {'activation': {'gelu'}, 'norm': {'layernorm'}, 'positional_encoding': {'sinusoidal'}})})
        gpu = hardware.get('gpu')
        gpu = gpu if gpu in {'NVIDIA GeForce RTX 3060', 'Tesla P100-PCIE-16GB', 'NVIDIA Tesla P100-PCIE-16GB', 'Tesla P100-SXM2-16GB'} else None
        system = metrics.get('system', {})
        tables['hardware'].append({**identity, 'gpu': gpu, 'dtype': hardware.get('dtype') if hardware.get('dtype') in {'float32', 'float16', 'bfloat16'} else None, 'python': version(system.get('python_version')), 'torch': version(system.get('torch_version')), 'cuda': version(hardware.get('cuda')), 'cuda_note': 'null means not independently recorded; torch build suffix retained'})
        tables['compression'].append({**identity, 'source_chars': train['source_chars'], 'source_bytes': train['source_bytes'], 'tokens': train['tokens'], 'chars_per_token': ratio(train['source_chars'], train['tokens']), 'bytes_per_token': ratio(train['source_bytes'], train['tokens'])})
        tables['conditions'].append({**identity, **numbers(config.get('controls', {}), CONTROLS), **enums(config.get('controls', {}), {'scheduler_axis': {'source_chars'}, 'regime': {'same_source_characters'}})})
        tables['provenance'].append({**identity, 'metrics_sha256': digest(source_hash), **{k: digest(metrics.get(k)) for k in ['dataset_sha256', 'records_sha256', 'tokenizer_sha256', 'checkpoint_sha256']}, **{f'code_{name}': digest(metrics.get('code_sha256', {}).get(name)) for name in ['phase55_same_source_runtime.py', 'probe_lm.py', 'probe_lm_data.py', 'tokenizer_adapters.py']}})
    for row in tables['compression']:
        for size in (32, 48):
            base = next((b for b in tables['compression'] if b['seed'] == row['seed'] and b['candidate'] == f'j-reversible-sp-unigram-{size}k' and b['source_chars'] == row['source_chars'] and b['source_bytes'] == row['source_bytes']), None)
            row[f'token_ratio_to_j{size}'] = ratio(row['tokens'], base['tokens']) if base else None
    return tables

def load_runs(payload, extras):
    runs = []
    if payload:
        with ZipFile(payload) as archive:
            for name in sorted(archive.namelist()):
                if name.startswith('results/cuda_same_source/runs/') and name.endswith('/artifacts/metrics.json'):
                    root = name.removesuffix('artifacts/metrics.json')
                    raw = archive.read(name)
                    runs.append((json.loads(raw), json.loads(archive.read(root + 'artifacts/config.json')), json.loads(archive.read(root + 'hardware.json')), hashlib.sha256(raw).hexdigest()))
    for root in extras:
        root = Path(root)
        raw = (root / 'artifacts/metrics.json').read_bytes()
        runs.append((json.loads(raw), json.loads((root / 'artifacts/config.json').read_text()), json.loads((root / 'hardware.json').read_text()) if (root / 'hardware.json').exists() else {}, hashlib.sha256(raw).hexdigest()))
    return runs

def export_observation(source, output):
    observed = json.loads(Path(source).read_text())
    timestamp = observed.get('observed_at')
    if not isinstance(timestamp, str) or not re.fullmatch(r'[0-9T:.+Z-]+', timestamp):
        raise ValueError('invalid observation timestamp')
    safe = {'observed_at': timestamp, 'scope': 'observed during J64 run; not retrospective proof of old runs',
            **{k: version(observed.get(k)) for k in ['python', 'pytorch', 'cuda_runtime', 'sentencepiece']},
            **numbers(observed, ['cudnn']), 'driver': [version(v) for v in observed.get('driver', [])], 'gpus': []}
    for gpu in observed.get('gpus', []):
        name = gpu.get('name')
        if name not in {'NVIDIA GeForce RTX 3060', 'Tesla P100-PCIE-16GB'}:
            raise ValueError('unknown GPU; review before publishing')
        cc = gpu.get('compute_capability')
        if not isinstance(cc, list) or len(cc) != 2 or any(type(x) is not int for x in cc):
            raise ValueError('invalid compute capability')
        safe['gpus'].append({'name': name, **numbers(gpu, ['total_memory_bytes']), 'compute_capability': cc})
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'software_observation.json').write_text(json.dumps(safe, indent=2) + '\n')


def export(payload, extras, output):
    tables = aggregate(load_runs(payload, extras))
    if not tables['summary']:
        raise ValueError('no measurements supplied')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (output / (name + '.json')).write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
        if rows:
            with (output / (name + '.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    return tables

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--extra-run', type=Path, action='append', default=[])
    parser.add_argument('--tokenizer-config-root', type=Path)
    parser.add_argument('--software-observation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tables = export(args.payload, args.extra_run, args.output)
    if args.tokenizer_config_root:
        export_tokenizers(args.tokenizer_config_root, args.output)
    if args.software_observation:
        export_observation(args.software_observation, args.output)
    print(json.dumps({name: len(rows) for name, rows in tables.items()}))
