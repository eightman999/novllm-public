from canonical_corpus.phase551_audit import audit, normalize, statistics


def row(i, text, work=None):
    return dict(id=i, text=text, category='kanbun', document_id=work)


def test_audit_boundaries_and_unknown():
    a = '甲乙丙丁戊己庚辛壬癸' * 6
    ev = [row('exact', '原文保持検査'), row('normalized', 'Ａ ＢＣＤＥ'),
          row('long', a), row('short', '甲乙'), row('absent', 'あいうえおかき')]
    tr = [row('t1', '原文保持検査'), row('t2', 'ABCDE'), row('t3', '前'+a+'後')]
    result, _ = audit(ev, tr, scope='fixture')
    assert result[0]['train_exact_match']
    assert not result[1]['train_exact_match'] and result[1]['train_normalized_exact_match']
    assert result[2]['train_long_substring_match']
    assert result[2]['max_similarity'] == 1
    assert result[3]['train_near_match'] is None
    assert result[4]['max_similarity'] == 0
    assert all(r['work_overlap'] == 'unknown' for r in result)
    assert ev[1]['text'] == 'Ａ ＢＣＤＥ'


def test_near_threshold_fixed():
    # 6 of 7 unique 5-grams remain after changing the last character.
    ev = [row('e', 'abcdefghijk')]
    r, _ = audit(ev, [row('t', 'abcdefghijZ')], scope='fixture')
    assert r[0]['max_similarity'] == 6/7
    assert r[0]['train_near_match'] is True
    assert r[0]['train_short_substring_match'] is False


def test_work_stats_and_duplicate():
    s = statistics([row('1','abc','p1'),row('2','abc','p2'),row('3','x')])['kanbun']
    assert s['known_works'] == 2 and s['unknown_work_documents'] == 1
    assert s['exact_duplicate_extra_records'] == 1


def test_jsonl_unicode_line_separators_are_not_record_boundaries(tmp_path):
    import json
    from canonical_corpus.phase551_audit import rows
    text = '原文\u2028保持\u2029検査'
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps(row('u',text),ensure_ascii=False)+'\n')
    assert list(rows(path))[0]['text'] == text
