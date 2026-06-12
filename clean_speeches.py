"""
Cassady Shoaff | PresidentSpeech 2026 Revamp
Unified text cleaner: extracts the president's speech from White House
transcript files. Replaces text_cleaner.py (Biden) and text_cleaner_trump.py.

Key changes from the 2021 versions:
  * Extraction instead of deletion for ALL presidents (the Biden script's
    deletion approach let untagged interviewer turns slip through).
  * Speaker turns are parsed by splitting on dialogue tags and keeping only
    the president's segments, so multi-line questions can't escape.
  * Files are decoded as UTF-8 with a cp1252 fallback. The raw files are a
    mix of both; decode(errors='replace') was turning every apostrophe,
    em-dash, and non-breaking space into U+FFFD.
  * One script, one code path. President-specific bits live in CONFIG.
  * A validation pass reports any leftover speaker-tag-shaped text so a
    missed tag shows up in the console instead of in the corpus.

Usage:
    python clean_speeches.py biden
    python clean_speeches.py trump --input-dir path/to/trump --output out.txt
"""
import argparse
import pathlib
import re
import sys

# ---------------------------------------------------------------- config ---

CONFIG = {
    "biden": {
        # subscription footer appended to whitehouse.gov transcripts
        "boilerplate": [
            r"We['’]ll be in touch with the latest information on how "
            r"President Biden and his administration are working for the "
            r"American people, as well as ways you can get involved and help "
            r"our country build back better\.",
        ],
    },
    "trump": {
        "boilerplate": [],
    },
}

# tags whose speech we keep (transcribers are inconsistent about labels)
PRESIDENT_TAGS = {
    "THE PRESIDENT",
    "PRESIDENT TRUMP",
    "PRESIDENT BIDEN",
}

# An ALL-CAPS run of 1+ words ending in a colon (THE PRESIDENT:, MS. TRUMP:,
# SECRETARY MNUCHIN:, Q:), or a bare reporter "Q". First word must be 2+
# chars so a stray capital letter can't match. The bare Q only needs trailing
# whitespace: questions can open with an em-dash or lowercase ("Q — the
# numbers...", "Q more details..."), and "Q." initials in names don't match
# because the period fails the lookahead.
SPEAKER_TAG = re.compile(
    r"(\b[A-Z][A-Z'’.\-]+(?:\s+[A-Z][A-Z'’.\-]*)*\s*:"
    r"|\bQ\b(?=\s))"
)

TIMESTAMP = re.compile(r"\d{1,2}:\d{2}\s*[AP]\.?M\.?\s*(?:[ECMP][SD]T)?")

METADATA = re.compile(r"<[^>]*>")          # <title="...">, <date="...">
PARENTHETICAL = re.compile(r"\([^()]*\)")  # (Applause.), (inaudible), ...
ADDRESS = re.compile(
    r"The White House\s+1600 Pennsylvania Ave NW\s+Washington, DC 20500"
)
ISSUED_HEADER = re.compile(r"^.*?Issued on:\s*\w+ \d{1,2}, \d{4}")

# ------------------------------------------------------------- pipeline ----


def load_text(path):
    """Decode a transcript, trying UTF-8 first, then cp1252.

    The Trump scrape is UTF-8 but the Biden scrape is cp1252 (0xA0
    non-breaking spaces, 0x92 apostrophes, ...). latin-1 is the never-fails
    last resort.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue


def preclean(text, boilerplate):
    """Remove markup and boilerplate, collapse all whitespace to spaces."""
    text = METADATA.sub("", text)
    text = PARENTHETICAL.sub("", text)
    # \s matches \xa0 once the file is decoded properly, so this also
    # normalizes the non-breaking spaces whitehouse.gov loves
    text = re.sub(r"\s+", " ", text)
    text = ADDRESS.sub("", text)
    for pattern in boilerplate:
        text = re.sub(pattern, "", text)
    return text


def normalize_tag(tag):
    return re.sub(r"\s+", " ", tag.rstrip(":").strip())


def extract_president(text):
    """Keep only segments spoken under a president dialogue tag.

    Splitting on every speaker tag (instead of deleting other speakers'
    lines) means anything not explicitly tagged as the president is
    dropped, so untagged or multi-line interviewer turns can't leak in.
    """
    parts = SPEAKER_TAG.split(text)
    # parts = [preamble, tag, segment, tag, segment, ...]; preamble is the
    # header before anyone speaks, so it is discarded
    kept = []
    for tag, segment in zip(parts[1::2], parts[2::2]):
        if normalize_tag(tag) in PRESIDENT_TAGS:
            kept.append(segment.strip())
    return " ".join(kept)


def clean_standalone(text):
    """Clean a speech with no dialogue tags (the whole body is the president).

    Everything up to the opening timestamp (e.g. "2:30 P.M. EDT") is header;
    files without a timestamp fall back to stripping the "Issued on:" header.
    """
    match = TIMESTAMP.search(text)
    if match:
        text = text[match.end():]
    else:
        text = ISSUED_HEADER.sub("", text)
    return text


def clean_file(path, boilerplate):
    """Returns (cleaned_text, mode) where mode is 'dialogue' or 'standalone'."""
    text = preclean(load_text(path), boilerplate)
    if "THE PRESIDENT:" in text or "PRESIDENT TRUMP:" in text \
            or "PRESIDENT BIDEN:" in text:
        text = TIMESTAMP.sub("", text)
        cleaned, mode = extract_president(text), "dialogue"
    else:
        cleaned, mode = clean_standalone(text), "standalone"
    cleaned = TIMESTAMP.sub("", cleaned)
    cleaned = re.sub(r"\bEND\b", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(), mode


def validate(corpus):
    """Report anything that still looks like a speaker tag."""
    leftovers = {}
    for match in SPEAKER_TAG.finditer(corpus):
        tag = normalize_tag(match.group())
        if tag not in leftovers:
            start = max(match.start() - 30, 0)
            leftovers[tag] = corpus[start:match.end() + 50]
    return leftovers


# ------------------------------------------------------------------ main ---


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("president", choices=sorted(CONFIG))
    parser.add_argument("--input-dir", default=None,
                        help="folder of raw transcripts (default: ./<president>)")
    parser.add_argument("--output", default=None,
                        help="output file (default: <president>_cleaned.txt)")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir or args.president)
    output_path = pathlib.Path(args.output or f"{args.president}_cleaned.txt")
    boilerplate = CONFIG[args.president]["boilerplate"]

    files = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and f"{args.president}_speeches" in p.name)
    if not files:
        sys.exit(f"No '{args.president}_speeches*' files found in {input_dir}")

    pieces, modes = [], {"dialogue": 0, "standalone": 0}
    for path in files:
        cleaned, mode = clean_file(path, boilerplate)
        modes[mode] += 1
        if cleaned:
            pieces.append(cleaned)

    corpus = " ".join(pieces)
    output_path.write_text(corpus, encoding="utf-8")

    print(f"Cleaned {len(files)} files "
          f"({modes['dialogue']} dialogue, {modes['standalone']} standalone)")
    print(f"Saved {output_path} ({len(corpus.split()):,} words)")

    leftovers = validate(corpus)
    if leftovers:
        print(f"\nWARNING: {len(leftovers)} possible uncleaned speaker tag(s):")
        for tag, context in sorted(leftovers.items()):
            print(f"  {tag!r}: ...{context}...")
    else:
        print("Validation: no speaker-tag-shaped text remains.")


if __name__ == "__main__":
    main()
