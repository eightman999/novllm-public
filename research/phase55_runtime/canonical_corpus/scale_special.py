"""Phase 4.6 lossless acquisition and audit for Kanbun-LM, JHPT and private AA.

This module deliberately writes only to a caller supplied Phase 4.6 run.  It
does not mutate the Phase 4.5 payloads and does not infer copyright status.
"""
from __future__ import annotations

import csv, hashlib, json, urllib.request, io, re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cultural_store import write_source_records, validate_source_store

KANBUN_BASE = "https://raw.githubusercontent.com/nlp-waseda/Kanbun-LM/main/kanbun-lm-dataset"
JHPT_API = "https://api.github.com/repos/nict-astrec-att/jhpt/contents/data02/bitext/sources"
JHPT_DIRS = {"Rekihaku": "CC-BY-4.0", "Fukui": "CC-BY-4.0", "SRHCA": "CC-BY-4.0", "Edo_Cooking": "CC-BY-SA-4.0"}

def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()

def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "novllm-phase46-audit/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()

def audit_kanbun(input_dir: str | Path, out: str | Path, *, fetch_official: bool = False) -> dict[str, Any]:
    """Import every row from train/val/test and retain the split verbatim."""
    input_dir, out = Path(input_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for split in ("train", "val", "test"):
        path = input_dir / f"{split}.csv"
        payload = _get(f"{KANBUN_BASE}/{split}.csv") if fetch_official else path.read_bytes()
        if fetch_official: path = out / "upstream" / f"{split}.csv"; path.parent.mkdir(exist_ok=True)
        if fetch_official: path.write_bytes(payload)
        raw = path.read_bytes(); files.append({"path": str(path), "split": split, "bytes": len(raw), "sha256": sha256(raw)})
        file_rows=[]
        for i, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))):
            required = ("poetry_id", "hakubun", "kakikudashi", "reading_order_ja")
            if any(k not in row for k in required): raise ValueError(f"Kanbun {split} row {i} missing field")
            # pandas-export index is intentionally retained as source metadata.
            rows.append({"record_id": f"kanbun_lm:{split}:{i}", "source_id": "kanbun_lm",
                         "text_raw": row["hakubun"], "hakubun": row["hakubun"],
                         "kakikudashi": row["kakikudashi"], "poetry_id": row["poetry_id"],
                         "reading_order_ja": row["reading_order_ja"], "split": split,
                         "metadata_raw": row, "rights_basis": "JP_Copyright_Act_30_4",
                         "redistribution": "not_allowed", "git_commit_allowed": False,
                         "storage_policy": "train_only"})
            file_rows.append(row)
        files[-1].update({"rows":len(file_rows), "characters":{"hakubun":sum(len(r["hakubun"]) for r in file_rows), "kakikudashi":sum(len(r["kakikudashi"]) for r in file_rows)},
                          "unique_poetry_ids":len({r["poetry_id"] for r in file_rows}), "url":f"{KANBUN_BASE}/{split}.csv", "revision":"main"})
    manifest = write_source_records(rows, out / "canonical", source_id="kanbun_lm", stage="raw")
    canonical_match = all(all(r["metadata_raw"].get(k) == r.get(k) for k in ("poetry_id","hakubun","kakikudashi","reading_order_ja")) for r in rows)
    comparison = validate_kanbun_export(input_dir if not fetch_official else out / "upstream", out / "canonical")
    result = {"source_id": "kanbun_lm", "origin_url": "https://github.com/nlp-waseda/Kanbun-LM",
              "files": files, "counts": {"rows": len(rows), "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}},
              "manifest": manifest, "upstream_canonical_field_match": canonical_match, "training_basis": "JP_Copyright_Act_30_4",
              "redistribution": "not_allowed", "git_commit_allowed": False, "storage_policy": "train_only",
              "validation": validate_source_store(out / "canonical", source_id="kanbun_lm"), "upstream_canonical_validation": comparison}
    (out / "kanbun_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result

def _jhpt_record(source_id: str, row: Mapping[str, Any], i: int, filename: str) -> dict[str, Any]:
    if "src_text" not in row or "ref_text" not in row: raise ValueError(f"JHPT {filename} row {i} missing src/ref")
    src, ref = row["src_text"], row["ref_text"]
    if not isinstance(src, str) or not isinstance(ref, str): raise ValueError("JHPT src/ref must preserve strings")
    return {"record_id": f"{source_id}:{row.get('id', i)}", "source_id": source_id,
            "text_raw": src, "src_text": src, "ref_text": ref, "commentary": row.get("commentary"),
            "ignore": row.get("ignore"), "metadata_raw": dict(row), "file": filename}

def audit_jhpt(source_root: str | Path, out: str | Path, *, fetch_all: bool = False) -> dict[str, Any]:
    """Audit all files in the four licensed data02 source directories.

    Empty strings, JSON null and ignore flags remain distinct fields.
    """
    source_root, out = Path(source_root), Path(out); out.mkdir(parents=True, exist_ok=True)
    summary, registries = {}, []
    for dirname, license_name in JHPT_DIRS.items():
        payloads: list[tuple[str, bytes]] = []
        if fetch_all:
            listing = json.loads(_get(f"{JHPT_API}/{dirname}"))
            for item in listing:
                if item.get("name", "").endswith(".jsonl"): payloads.append((item["name"], _get(item["download_url"])))
        else:
            local = source_root / dirname
            if local.is_dir(): payloads = [(p.name, p.read_bytes()) for p in sorted(local.glob("*.jsonl"))]
            else:
                # Phase 4.5 uses flat names; import those without changing them.
                alias = {"Rekihaku": "rekihaku", "Fukui": "fukui", "SRHCA": "srhca", "Edo_Cooking": "edo_cooking"}[dirname]
                p = source_root / f"{alias}.jsonl"
                if p.exists(): payloads = [(p.name, p.read_bytes())]
        source_id = "jhpt_" + dirname.lower()
        records, files = [], []
        for filename, payload in payloads:
            target = out / "upstream" / dirname / filename
            if fetch_all: target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
            files.append({"filename": filename, "bytes": len(payload), "sha256": sha256(payload)})
            file_records=[]
            for i, line in enumerate(payload.splitlines()):
                if not line.strip(): continue
                rec=_jhpt_record(source_id, json.loads(line), i, filename); records.append(rec); file_records.append(rec)
            files[-1]["rows"]=len(file_records)
        manifest = write_source_records(records, out / "canonical", source_id=source_id, stage="raw")
        src_records = [r for r in records if not r["ignore"]]
        def official_src(s): return re.sub(r"《[^》]*》", "", s)
        def official_ref(s): return re.sub(r"（[^）]*）", "", re.sub(r"《[^》]*》", "", s))
        comparison = validate_jhpt_export((out / "upstream" / dirname) if fetch_all else source_root, out / "canonical", source_id, dirname)
        summary[source_id] = {"source_dir": dirname, "license": license_name, "files": files,
                              "records": len(records), "usable_records_ignore_false": len(src_records),
                              "src_chars": sum(len(r["src_text"]) for r in records),
                              "ref_chars": sum(len(r["ref_text"]) for r in records), "manifest": manifest,
                              "validation": validate_source_store(out / "canonical", source_id=source_id), "upstream_canonical_validation": comparison,
                              "official_definition_counts": {"usable_records":len(src_records), "src_chars":sum(len(official_src(r["src_text"]) ) for r in src_records), "ref_chars":sum(len(official_ref(r["ref_text"]) ) for r in src_records)}}
        registries.append({"source_id": source_id, "source_type": "jhpt", "license": license_name,
                           "training_basis": "open_license", "redistribution": "allowed_with_conditions",
                           "git_commit_allowed": False, "storage_policy": "licensed_local",
                           "origin_url": f"https://github.com/nict-astrec-att/jhpt/tree/main/data02/bitext/sources/{dirname}",
                           "files": files})
    result = {"origin_url": "https://github.com/nict-astrec-att/jhpt", "sources": summary,
              "reported_reference": {"documents": 30, "segments": 543, "orig": 22744, "ref": 27220,
                                     "definition": "ruby annotations excluded"},
              "comparison_note": "raw counts include title/ignored/empty/null values; official counts remove ruby 《》 and reference supplemental （） only; canonical raw is unchanged",
              "registry": registries}
    official={"documents":sum(len(v["files"]) for v in summary.values()),"usable_segments":sum(v["official_definition_counts"]["usable_records"] for v in summary.values()),"orig_chars":sum(v["official_definition_counts"]["src_chars"] for v in summary.values()),"ref_chars":sum(v["official_definition_counts"]["ref_chars"] for v in summary.values())}
    result["official_metadata_match"] = official == {"documents":30,"usable_segments":543,"orig_chars":22744,"ref_chars":27220}; result["official_counts"] = official
    (out / "jhpt_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result

def validate_kanbun_export(input_dir: str|Path, canonical_root: str|Path, expected_files: Mapping[str,Any]|None=None) -> dict[str,Any]:
    """Reload canonical JSONL and compare every upstream CSV field exactly."""
    expected=[]; input_dir=Path(input_dir)
    for split in ("train","val","test"):
        p=input_dir/f"{split}.csv"
        if not p.exists(): return {"valid":False,"error":f"missing:{p}"}
        raw_bytes=p.read_bytes()
        if expected_files and split in expected_files and expected_files[split].get("sha256") != sha256(raw_bytes): return {"valid":False,"error":"input_sha_mismatch","split":split}
        for row in csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig"), newline="")):
            expected.append((split,{k:row[k] for k in ("poetry_id","hakubun","kakikudashi","reading_order_ja")}))
    try:
        from .cultural_store import iter_source_records
        actual=list(iter_source_records(canonical_root, source_id="kanbun_lm")); strict=validate_source_store(canonical_root,source_id="kanbun_lm")
    except Exception as e: return {"valid":False,"error":str(e)}
    if len(expected)!=len(actual): return {"valid":False,"error":"row_count","expected":len(expected),"actual":len(actual)}
    if not strict.get("valid"): return {"valid":False,"error":"strict_store_invalid"}
    for i,((split,fields),row) in enumerate(zip(expected,actual)):
        got={k:row.get("metadata_raw",{}).get(k) for k in fields}
        tops={k:row.get(k) for k in fields}; tops["split"]=row.get("split")
        if got!=fields or any(tops.get(k)!=v for k,v in fields.items()) or row.get("split")!=split or row.get("text_raw")!=fields["hakubun"]: return {"valid":False,"error":"field_mismatch","row":i}
    return {"valid":True,"rows":len(actual),"fields":["poetry_id","hakubun","kakikudashi","reading_order_ja","split"]}

def validate_jhpt_export(source_root: str|Path, canonical_root: str|Path, source_id: str, dirname: str, expected_files: Mapping[str,Any]|None=None) -> dict[str,Any]:
    """Reload each JHPT file and compare locator, file hash, and all JSON fields."""
    source_root=Path(source_root); files=sorted(source_root.glob("*.jsonl")) if source_root.is_dir() else []
    if not files:
        alias={"Rekihaku":"rekihaku","Fukui":"fukui","SRHCA":"srhca","Edo_Cooking":"edo_cooking"}.get(dirname)
        if alias and (source_root/f"{alias}.jsonl").exists(): files=[source_root/f"{alias}.jsonl"]
    expected=[]
    for p in files:
        digest=sha256(p.read_bytes())
        if expected_files and p.name in expected_files and expected_files[p.name].get("sha256") != digest: return {"valid":False,"error":"input_sha_mismatch","file":p.name}
        for i,line in enumerate(p.read_bytes().splitlines()):
            if line.strip(): expected.append((p.name,i,json.loads(line),digest))
    try:
        from .cultural_store import iter_source_records
        actual=list(iter_source_records(canonical_root, source_id=source_id)); strict=validate_source_store(canonical_root,source_id=source_id)
    except Exception as e: return {"valid":False,"error":str(e)}
    if not strict.get("valid"): return {"valid":False,"error":"strict_store_invalid"}
    if not expected: return {"valid":False,"error":"empty_source_files"}
    if len(expected)!=len(actual): return {"valid":False,"error":"row_count","expected":len(expected),"actual":len(actual)}
    for i,((name,locator,raw,digest),row) in enumerate(zip(expected,actual)):
        if row.get("file")!=name or row.get("metadata_raw")!=raw or row.get("src_text")!=raw.get("src_text") or row.get("ref_text")!=raw.get("ref_text") or row.get("text_raw")!=raw.get("src_text") or row.get("commentary")!=raw.get("commentary") or row.get("ignore")!=raw.get("ignore"): return {"valid":False,"error":"field_or_locator_mismatch","row":i,"file":name}
    return {"valid":True,"rows":len(actual),"files":len(files),"file_sha256":{p.name:sha256(p.read_bytes()) for p in files}}

def sample_private_aa(path: str | Path, out: str | Path, *, min_chars: int = 12_000_000) -> dict[str, Any]:
    """Copy heuristic aa_candidate rows as metadata-separated, exact-text raw records."""
    rows, chars = [], 0
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line); strata = row.get("provenance", {}).get("strata", {})
            if not strata.get("aa_candidate"): continue
            text = row.get("text", ""); chars += len(text)
            episode_id = row["boundary"]["episode_revision_id"]
            provenance = row.get("provenance", {})
            rows.append({"record_id": episode_id, "source_id": "aa_private",
                         "text_raw": text, "episode_revision_id": row["boundary"]["episode_revision_id"],
                         "raw_source_episode_id": episode_id, "sample_text_sha256": sha256(text),
                         "stable_work_key": row["boundary"]["stable_work_key"], "rawhash": provenance.get("raw_sha256"),
                         "parent_stage":"phase3-tokenizer-derived", "parent_recipe":provenance.get("recipe"),
                         "parent_text_sha256":provenance.get("text_sha256"), "parent_provenance":provenance,
                         "label": "heuristic_aa_candidate", "aa_label_confirmed": False, "heuristic": True,
                         "whitespace_exact": True, "training_basis": "userpolicy30_4", "redistribution": "not_allowed",
                         "git_commit_allowed": False, "storage_policy": "private"})
    if chars < min_chars: raise ValueError(f"AA candidate chars {chars} < {min_chars}")
    root = Path(out); manifest = write_source_records(rows, root / "canonical", source_id="aa_private", stage="raw")
    result = {"source_id": "aa_private", "records": len(rows), "characters": chars, "manifest": manifest,
              "selector": "provenance.strata.aa_candidate == true", "label_status": "heuristic_only",
              "rights_basis": "user research policy; no legal-rights inference", "training_basis": "userpolicy30_4",
              "exclusion_key_fields": ["episode_revision_id", "stable_work_key", "rawhash", "sample_text_sha256"], "read_only_source": str(path)}
    (root / "aa_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result

def build_exclusion_inventory(aa_rows: Iterable[Mapping[str, Any]], web_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare AA and ordinary web pools by stable episode ID and text hash."""
    def keys(rows):
        ids, hashes = set(), set()
        for row in rows:
            b = row.get("boundary", row)
            eid = b.get("episode_revision_id") or row.get("raw_source_episode_id")
            text = row.get("text", row.get("text_raw", ""))
            if eid: ids.add(eid)
            hashes.add(row.get("sample_text_sha256") or row.get("text_sha256") or sha256(text))
        return ids, hashes
    aa_ids, aa_hashes = keys(aa_rows); web_ids, web_hashes = keys(web_rows)
    return {"aa_episode_count": len(aa_ids), "aa_text_count": len(aa_hashes),
            "web_episode_count": len(web_ids), "web_text_count": len(web_hashes),
            "episode_id_intersection": sorted(aa_ids & web_ids),
            "text_sha256_intersection": sorted(aa_hashes & web_hashes),
            "disjoint": not (aa_ids & web_ids or aa_hashes & web_hashes),
            "key_fields": ["episode_revision_id", "sample_text_sha256"]}

def select_non_aa_web(rows: Iterable[Mapping[str, Any]], exclusion: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Selector usable against a materialized Phase 2/3 pool without mutation."""
    ids = set(exclusion.get("episode_ids", exclusion.get("episode_revision_ids", [])))
    hashes = set(exclusion.get("text_sha256", exclusion.get("sample_text_sha256", [])))
    out = []
    for row in rows:
        b = row.get("boundary", row); eid = b.get("episode_revision_id") or row.get("raw_source_episode_id")
        text = row.get("text", row.get("text_raw", "")); digest = row.get("sample_text_sha256") or row.get("text_sha256") or sha256(text)
        if eid not in ids and digest not in hashes: out.append(row)
    return out

__all__ = ["audit_kanbun", "audit_jhpt", "sample_private_aa", "build_exclusion_inventory", "select_non_aa_web", "sha256"]
