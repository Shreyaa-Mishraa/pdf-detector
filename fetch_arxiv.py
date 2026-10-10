"""Download benign PDFs from arXiv and record each paper's ID and license.

Metadata (including the license) comes from arXiv's OAI-PMH interface; PDFs
come from export.arxiv.org, the mirror arXiv asks automated clients to use.
Following arXiv's guidance for automated access, it waits 3 seconds between
requests. Papers with no license in their metadata are skipped.

Output:
  data/raw/benign/<arxiv_id>.pdf      ("/" in old-style IDs becomes "_")
  data/arxiv_sources.csv              filename, arxiv_id, license (appended)

Re-running resumes: papers already listed in arxiv_sources.csv are skipped.
"""

import argparse
import csv
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OAI_URL = "https://oaipmh.arxiv.org/oai"
PDF_URL = "https://export.arxiv.org/pdf/{}"
DELAY = 3.0
USER_AGENT = "pdf-detector-dataset-builder/1.0 (academic project)"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/",
      "ax": "http://arxiv.org/OAI/arXiv/"}


def http_get(url):
    """GET with a polite delay; honours 503 Retry-After from OAI-PMH."""
    for _ in range(5):
        time.sleep(DELAY)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                time.sleep(int(e.headers.get("Retry-After", 30)))
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def list_records(oai_set, date_from):
    """Yield (arxiv_id, license) for records in the set, page by page."""
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv",
              "set": oai_set, "from": date_from}
    while True:
        root = ET.fromstring(http_get(OAI_URL + "?" + urllib.parse.urlencode(params)))
        for meta in root.iterfind(".//oai:record/oai:metadata/ax:arXiv", NS):
            yield meta.findtext("ax:id", "", NS).strip(), meta.findtext("ax:license", "", NS).strip()
        token = root.findtext(".//oai:resumptionToken", "", NS).strip()
        if not token:
            return
        params = {"verb": "ListRecords", "resumptionToken": token}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of PDFs to collect")
    ap.add_argument("--set", default="cs", help="OAI-PMH set, e.g. cs, math, physics")
    ap.add_argument("--from", dest="date_from", default="2024-06-01",
                    help="only records updated on/after this date (YYYY-MM-DD)")
    ap.add_argument("--max-mb", type=float, default=10.0,
                    help="skip PDFs larger than this (the web app rejects >10 MB)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "benign")
    ap.add_argument("--sources", type=Path, default=ROOT / "data" / "arxiv_sources.csv")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.sources.exists():
        with open(args.sources, newline="", encoding="utf-8") as f:
            done = {row["arxiv_id"] for row in csv.DictReader(f)}
    new_file = not args.sources.exists() or args.sources.stat().st_size == 0

    with open(args.sources, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filename", "arxiv_id", "license"])
        got = len(done)
        for arxiv_id, license_url in list_records(args.set, args.date_from):
            if got >= args.n:
                break
            if not arxiv_id or arxiv_id in done:
                continue
            if not license_url:
                print(f"skip {arxiv_id}: no license in metadata")
                continue
            try:
                data = http_get(PDF_URL.format(arxiv_id))
            except Exception as e:
                print(f"skip {arxiv_id}: download failed ({e})")
                continue
            if not data.startswith(b"%PDF"):
                print(f"skip {arxiv_id}: not a PDF (withdrawn or source-only?)")
                continue
            if len(data) > args.max_mb * 1024 * 1024:
                print(f"skip {arxiv_id}: {len(data) / 1e6:.1f} MB > {args.max_mb} MB")
                continue
            name = arxiv_id.replace("/", "_") + ".pdf"
            (args.out_dir / name).write_bytes(data)
            w.writerow([name, arxiv_id, license_url])
            f.flush()
            done.add(arxiv_id)
            got += 1
            print(f"[{got}/{args.n}] {arxiv_id}  {license_url}")


if __name__ == "__main__":
    main()
