"""Build data/manifest.csv from the raw sample folders.

Run this INSIDE the isolated, network-less VM/container that holds the samples.
It only reads raw bytes to hash them; it never opens or renders a PDF.

Steps:
  1. SHA-256 every file in data/raw/benign and data/raw/malicious.
  2. Drop exact duplicates (and any hash that appears in both classes).
  3. Balance the classes by randomly down-sampling the larger one.
  4. Stratified train/test split with a fixed seed.
  5. Write data/manifest.csv: sha256, filename, label, split, source.

Labels: 1 = malicious, 0 = benign.
"source" is "contagio" for malicious files and "arxiv:<id>" for benign files
(looked up in data/arxiv_sources.csv, which also records each paper's license).
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
FIELDS = ["sha256", "filename", "label", "split", "source"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_arxiv_ids(path):
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        # Only rows with both an ID and a license count as recorded.
        return {row["filename"]: row["arxiv_id"].strip()
                for row in csv.DictReader(f)
                if row["arxiv_id"].strip() and row["license"].strip()}


def hash_folder(folder):
    """Return {sha256: relative filename}, keeping the first file seen per hash."""
    seen, dupes = {}, 0
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        digest = sha256_of(p)
        if digest in seen:
            dupes += 1
            continue
        seen[digest] = p.relative_to(RAW).as_posix()
    return seen, dupes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "manifest.csv")
    args = ap.parse_args()

    benign, b_dupes = hash_folder(RAW / "benign")
    malicious, m_dupes = hash_folder(RAW / "malicious")
    overlap = set(benign) & set(malicious)
    for h in overlap:
        del benign[h], malicious[h]

    print(f"benign:    {len(benign)} unique ({b_dupes} duplicates removed)")
    print(f"malicious: {len(malicious)} unique ({m_dupes} duplicates removed)")
    print(f"hashes in both classes (dropped): {len(overlap)}")

    rng = random.Random(args.seed)
    n = min(len(benign), len(malicious))
    arxiv_ids = load_arxiv_ids(ROOT / "data" / "arxiv_sources.csv")

    rows = []
    for label, files in ((0, benign), (1, malicious)):
        items = sorted(files.items())
        rng.shuffle(items)
        items = items[:n]
        n_test = round(n * args.test_size)
        for i, (digest, name) in enumerate(items):
            if label == 1:
                source = "contagio"
            else:
                arxiv_id = arxiv_ids.get(Path(name).name)
                if arxiv_id is None:
                    raise SystemExit(f"{name} has no arXiv ID + license in data/arxiv_sources.csv "
                                     "(every benign file needs an arXiv ID and license)")
                source = "arxiv:" + arxiv_id
            rows.append({
                "sha256": digest,
                "filename": name,
                "label": label,
                "split": "test" if i < n_test else "train",
                "source": source,
            })

    rows.sort(key=lambda r: (r["split"], r["label"], r["sha256"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    for split in ("train", "test"):
        b = sum(r["split"] == split and r["label"] == 0 for r in rows)
        m = sum(r["split"] == split and r["label"] == 1 for r in rows)
        print(f"{split:5s}: {b} benign + {m} malicious = {b + m}")
    print(f"balanced to {n} per class; wrote {args.out}")


if __name__ == "__main__":
    main()
