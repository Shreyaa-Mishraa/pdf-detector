"""Extract PDF samples from password-protected Contagio zip archives.

Run this ONLY inside the isolated VM, with networking disabled.

Each member is read as bytes and written to data/raw/malicious/<sha256>
with no file extension, so nothing on the VM will try to open it as a PDF.
Members that have no %PDF header in their first 1024 bytes are skipped.
Archive member names are never used as paths, so path traversal is not possible.

Python's zipfile only handles classic ZipCrypto. For AES-encrypted zips or
.7z archives, extract with 7-Zip into an empty folder instead and pass that
folder with --from-dir.
"""

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw" / "malicious"


def save(data, stats):
    if b"%PDF" not in data[:1024]:
        stats["not_pdf"] += 1
        return
    digest = hashlib.sha256(data).hexdigest()
    target = OUT / digest
    if target.exists():
        stats["duplicate"] += 1
        return
    target.write_bytes(data)
    stats["saved"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archives", nargs="*", type=Path, help="zip archives")
    ap.add_argument("--password", help="archive password from the Contagio post")
    ap.add_argument("--from-dir", type=Path, help="folder already extracted with 7-Zip")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"saved": 0, "duplicate": 0, "not_pdf": 0}
    pwd = args.password.encode() if args.password else None

    for archive in args.archives:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                try:
                    save(z.read(info, pwd=pwd), stats)
                except NotImplementedError:
                    raise SystemExit(f"{archive}: unsupported encryption; use 7-Zip and --from-dir")
                except RuntimeError as e:
                    raise SystemExit(f"{archive}: {e} (wrong password?)")

    if args.from_dir:
        for p in sorted(args.from_dir.rglob("*")):
            if p.is_file():
                save(p.read_bytes(), stats)

    print(f"saved {stats['saved']}, duplicates {stats['duplicate']}, "
          f"skipped non-PDF {stats['not_pdf']} -> {OUT}")


if __name__ == "__main__":
    main()
