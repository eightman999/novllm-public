# export_novel_jsonl.py
import sqlite3, json, re, hashlib, os

SRC_DB = "./data/novel_status.db"  # DBの場所
OUT_DIR = "./data"
os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    """HTML除去・空白整形など軽めの掃除"""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def hash_norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

conn = sqlite3.connect(SRC_DB)
cur = conn.cursor()

# 小説リストを取得
cur.execute("SELECT n_code, title, author, main_tag, sub_tag FROM novels_descs")
novels = cur.fetchall()
total_written = 0
for n_code, title, author, main_tag, sub_tag in novels:
    cur.execute("SELECT body FROM episodes WHERE ncode = ? ORDER BY CAST(episode_no AS INTEGER)", (n_code,))
    bodies = [clean_text(b[0]) for b in cur.fetchall() if b[0]]
    if not bodies:
        continue

    # 全エピソードを連結して1テキスト化
    joined = "\n\n".join(bodies)
    if len(joined) < 500:
        continue  # 短すぎるのはスキップ

    out_path = os.path.join(OUT_DIR, f"{n_code}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        record = {
            "meta": {
                "ncode": n_code,
                "title": title,
                "author": author,
                "main_tag": main_tag,
                "sub_tag": sub_tag
            },
            "text": joined
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_written += 1
        print(f"✅ wrote {out_path}")

print(f"\nTotal novels exported: {total_written}")
conn.close()
