# PresidentSpeech — 2026 Revamp

A modernization of [cassadycodes/PresidentSpeech](https://github.com/cassadycodes/PresidentSpeech)
(APLN580 Corpus Linguistics, Spring 2021): cleaning White House transcripts down
to only the president's own speech, then comparing the Biden and Trump corpora.

## What changed since 2021

The original per-president cleaners had three significant bugs, all fixed here:

1. **The Biden cleaner deleted ~31% of Biden's own words.** Its speaker-tag
   filter `[A-Z]+\s` matched any paragraph starting with a single capital —
   so every paragraph beginning with "I ..." (or any acronym) was silently
   dropped. A biased loss for corpus work, since it skewed first-person usage.
2. **The Trump cleaner leaked other speakers.** Files tagged
   `PRESIDENT TRUMP:` instead of `THE PRESIDENT:` fell through to the
   no-dialogue branch, keeping entire turns from Prime Minister Varadkar,
   President Duque, and reporters whose `Q` opened with an em-dash.
3. **Encoding corruption.** The Biden scrape is cp1252 and the Trump scrape is
   UTF-8; decoding with `errors='replace'` destroyed apostrophes, em-dashes,
   and the non-breaking spaces the speaker-tag regexes relied on.

Because of 1 and 2, the cleaned corpora and figures in the original repo are
not comparable with the ones here.

## Contents

| Path | Description |
| --- | --- |
| `clean_speeches.py` | Unified extraction-based cleaner (`python clean_speeches.py biden`) with a validation pass that reports anything speaker-tag-shaped left in the output |
| `make_visuals.py` | All charts from both corpora in one run (`python make_visuals.py`) |
| `biden/`, `trump/` | Raw scraped transcripts (from the original repo's zips) |
| `*_cleaned.txt` | Re-cleaned corpora: Biden 111,690 words, Trump 361,770 |
| `visuals/` | Generated PNGs |

## Visuals

All frequency charts are per 10,000 content tokens so the two corpora compare
fairly despite the ~3x size difference (the originals plotted raw counts).

- **top20_words / top_bigrams / top_trigrams** — side-by-side panels
- **distinctive_words** — keyness (Dunning log-likelihood): each president's
  signature vocabulary relative to the other
- **lexical_diversity** — standardized TTR over 1,000-token windows, replacing
  raw TTR (which shrinks as a corpus grows and can't compare corpus sizes)

Known caveat: concatenating a president's dialogue turns can manufacture
cross-turn n-grams (e.g. "go ahead go ahead" from consecutive short answers).

## Requirements

Python 3.10+ with `matplotlib` and `nltk`
(`python -m nltk.downloader stopwords punkt_tab`).
