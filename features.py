"""Static feature extraction for PDF files.

Nothing here executes, renders or interprets PDF content. Keyword, object and
stream counts come from regexes over the raw bytes. pikepdf (qpdf) is used only
to check that the structure parses and to count pages, with every call wrapped
in try/except so malformed files never crash the caller.
"""

import re
import sys

import numpy as np
import pikepdf

# PDFiD-style keywords, counted as exact PDF name tokens.
KEYWORDS = ["/JavaScript", "/JS", "/OpenAction", "/Launch", "/EmbeddedFile",
            "/AA", "/URI", "/XFA", "/ObjStm", "/Encrypt"]

FEATURE_NAMES = KEYWORDS + ["file_size", "n_objects", "n_pages", "n_streams",
                            "parsed_ok", "entropy"]

# A PDF name is "/" followed by any run of non-delimiter, non-whitespace bytes.
NAME_RE = re.compile(rb"/[^\x00\t\n\x0c\r ()<>\[\]{}/%]*")
HEX_ESCAPE_RE = re.compile(rb"#([0-9A-Fa-f]{2})")
# Same rule as PDFiD: "obj"/"stream" not glued to other letters, so "endobj"
# and "endstream" are not counted.
OBJ_RE = re.compile(rb"(?<![A-Za-z])obj(?![A-Za-z])")
STREAM_RE = re.compile(rb"(?<![A-Za-z])stream(?![A-Za-z])")


def _normalize_name(name):
    """Decode #xx hex escapes, so /J#61vaScript counts as /JavaScript."""
    return HEX_ESCAPE_RE.sub(lambda m: bytes([int(m.group(1), 16)]), name)


def keyword_counts(data):
    counts = dict.fromkeys(KEYWORDS, 0)
    for match in NAME_RE.finditer(data):
        name = _normalize_name(match.group()).decode("latin-1")
        if name in counts:
            counts[name] += 1
    return counts


def byte_entropy(data):
    """Shannon entropy of the raw bytes, in bits per byte (0 to 8)."""
    if not data:
        return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    p = freq[freq > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def structure_info(path):
    """Return (parsed_ok, n_pages). n_pages is -1 when it cannot be read.

    qpdf silently repairs damaged files, so a file only counts as parsed_ok
    if it opened and qpdf raised no warnings while reading the page tree.
    """
    try:
        pdf = pikepdf.open(path)
    except Exception:
        return 0, -1
    try:
        try:
            n_pages = len(pdf.pages)
        except Exception:
            return 0, -1
        try:
            clean = not pdf.get_warnings()
        except Exception:
            clean = False
        return int(clean), n_pages
    finally:
        try:
            pdf.close()
        except Exception:
            pass


def extract_features(path):
    """Return a dict with exactly the keys in FEATURE_NAMES, in that order."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        data = b""

    parsed_ok, n_pages = structure_info(path)
    feats = keyword_counts(data)
    feats.update({
        "file_size": len(data),
        "n_objects": len(OBJ_RE.findall(data)),
        "n_pages": n_pages,
        "n_streams": len(STREAM_RE.findall(data)),
        "parsed_ok": parsed_ok,
        "entropy": round(byte_entropy(data), 6),
    })
    return {k: feats[k] for k in FEATURE_NAMES}


def to_vector(feats):
    return [float(feats[k]) for k in FEATURE_NAMES]


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, extract_features(p))
