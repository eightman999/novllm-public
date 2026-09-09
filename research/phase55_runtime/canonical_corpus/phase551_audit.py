"""Fixed-threshold, text-private Phase 5.5.1 contamination audit.

Normalization is an audit view only. It never changes evaluation text.
Near similarity is character 5-gram containment, not paraphrase equivalence.
"""
from __future__ import annotations
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

METHOD = {'normalization': 'NFKC then remove Unicode whitespace', 'ngram_n': 5,
          'near_metric': 'directional unique character 5-gram containment in one train record',
          'near_threshold': 0.8, 'long_substring_chars': 50,
          'short_substring_min_chars': 5,
          'fixed_before_audit': True}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if not c.isspace())


def grams(text, n=5):
    return {text[i:i+n] for i in range(len(text)-n+1)}


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def audit(evaluation, training, *, scope):
    """Stream train records; retain only eval-indexed overlaps and no train text.

    max_similarity scans every record with an overlapping n-gram: omitted
    records provably have zero similarity. Short (<5 normalized chars) items
    have null near metrics, never an asserted zero or clean certification.
    """
    ev = list(evaluation)
    norms = [normalize(r['text']) for r in ev]
    sets = [grams(t) for t in norms]
    index = defaultdict(list)
    exact_index, norm_index = defaultdict(list), defaultdict(list)
    for i, (r, ng) in enumerate(zip(ev, sets)):
        exact_index[sha(r['text'])].append(i)
        norm_index[sha(norms[i])].append(i)
        for g in ng:
            index[g].append(i)
    long_index = defaultdict(list)
    for i, text in enumerate(norms):
        for span in grams(text, 50):
            long_index[span].append(i)
    results = []
    for i, r in enumerate(ev):
        work = r.get('document_id')
        results.append({'eval_item_id': r['id'], 'category': r['category'],
            'source_work_identity': work or 'unknown', 'comparison_scope': scope,
            'train_exact_match': False, 'train_normalized_exact_match': False,
            'train_long_substring_match': False if len(norms[i]) >= 50 else None,
            'train_short_substring_match': False if len(norms[i]) >= 5 else None,
            'train_near_match': False if sets[i] else None,
            'max_similarity': 0.0 if sets[i] else None, 'matched_train_id': None,
            'work_overlap': 'unknown', 'method': METHOD['near_metric'],
            'threshold': METHOD['near_threshold'], 'status': 'pending'})
    train_count, train_chars = 0, 0
    known_works = set()
    unknown_works = 0
    for t in training:
        train_count += 1
        text, tid = t['text'], t['id']
        train_chars += len(text)
        if t.get('document_id'):
            known_works.add(t['document_id'])
        else:
            unknown_works += 1
        nt = normalize(text)
        for i in exact_index.get(sha(text), []):
            results[i]['train_exact_match'] = True
            results[i]['exact_matched_train_id'] = tid
        for i in norm_index.get(sha(nt), []):
            results[i]['train_normalized_exact_match'] = True
            results[i]['normalized_matched_train_id'] = tid
        for p in range(len(nt)-49):
            for i in long_index.get(nt[p:p+50], ()):
                results[i]["train_long_substring_match"] = True
                results[i]["long_substring_matched_train_id"] = tid
        counts = Counter()
        for g in grams(nt):
            counts.update(index.get(g, ()))
        for i, count in counts.items():
            result = results[i]
            similarity = count / len(sets[i])
            if similarity > result['max_similarity']:
                result['max_similarity'] = similarity
                result['matched_train_id'] = tid
            if similarity >= METHOD['near_threshold']:
                result['train_near_match'] = True
            if len(norms[i]) >= 5 and norms[i] in nt:
                result['train_short_substring_match'] = True
                result['substring_matched_train_id'] = tid
    for r in results:
        work = r['source_work_identity']
        r['work_overlap'] = True if work in known_works else 'unknown'
        # Different corpora's identity namespaces are not comparable. Absence
        # from an ID set is not proof that the underlying literary work differs.
        matched = any(r[k] is True for k in ('train_exact_match', 'train_normalized_exact_match',
                      'train_long_substring_match', 'train_short_substring_match', 'train_near_match'))
        r['status'] = 'match_detected' if matched else 'no_detected_text_match_work_unknown'
        if r['train_near_match'] is None and not matched:
            r['status'] = 'insufficient_length_work_unknown'
    return results, {'scope': scope, 'train_records': train_count, 'train_chars': train_chars,
                     'train_records_with_unknown_work_identity': unknown_works,
                     'method': METHOD, 'work_identity_absence_is_not_disjointness': True}


def statistics(evaluation):
    out = {}
    for category in sorted({r['category'] for r in evaluation}):
        selected = [r for r in evaluation if r['category'] == category]
        by_work = Counter()
        for r in selected:
            by_work[r.get('document_id') or 'unknown'] += len(r['text'])
        chars = sum(len(r['text']) for r in selected)
        out[category] = {'documents': len(selected), 'characters': chars,
            'bytes': sum(len(r['text'].encode()) for r in selected),
            'known_works': len(set(by_work)-{'unknown'}),
            'unknown_work_documents': sum(not r.get('document_id') for r in selected),
            'max_work_character_fraction': max(by_work.values(), default=0)/chars if chars else None,
            'work_characters': dict(sorted(by_work.items())),
            'exact_duplicate_extra_records': len(selected)-len({sha(r['text']) for r in selected})}
    return out


def kanbun_candidates(canonical):
    """Preserve every source line and its original split; never join poem lines."""
    out = []
    for r in canonical:
        for field, category in [('hakubun', 'kanbun'), ('kakikudashi', 'kakikudashi')]:
            text = r[field]
            if not text:
                continue
            out.append({'id': f'phase551:{category}:{r["record_id"]}',
                'document_id': f'kanbun_lm:poetry:{r["poetry_id"]}',
                'category': category, 'text': text, 'text_sha256': sha(text),
                'source': 'https://github.com/nlp-waseda/Kanbun-LM',
                'source_record_id': r['record_id'], 'upstream_split': r['split'],
                'source_chars': len(text), 'source_bytes': len(text.encode()),
                'redistribution': 'not_allowed', 'git_commit_allowed': False,
                'segmentation': 'one_original_source_row_no_join'})
    return out
