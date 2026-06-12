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
- **`*_substantive` variants** — the same four charts with an extended
  stoplist (light verbs, discourse markers, generic time/evaluative words),
  surfacing topical content instead of style. Substantive n-grams are formed
  over the standard token stream and then filtered, so removing a word can't
  join two words that were never adjacent ("great job, great job" must not
  become "job job"). Style words aren't lost — the standard-stoplist keyness
  chart is where they show up.
- **top20_lemmas** — frequencies of word *families*: POS-aware WordNet
  lemmatization (said → say) plus Snowball stemming of the lemma merges
  inflections and derivations (vaccine / vaccinated / vaccination count as
  one item), with each family labeled by its most frequent surface form.
  Uses the extended stoplist.
- **top20_verbs** — most frequent lexical verbs, POS-tagged in sentence
  context, inflections merged. Auxiliaries, modals, politeness formulae, and
  bleached constructions (future "going to", habitual "used to") are
  excluded; tagging in context is what keeps genuine "going" (motion) and
  "use" (instrumental) while dropping their filler uses.
- **lexical_diversity** — standardized TTR over 1,000-token windows, replacing
  raw TTR (which shrinks as a corpus grows and can't compare corpus sizes)
- **sentiment** — per-sentence VADER polarity: share of positive / neutral /
  negative sentences and the score distribution. Read with care: VADER scores
  evaluative *language*, so it measures how upbeat the wording is, not whether
  the news delivered is good.

Before any analysis, sentences under 4 words are dropped (9% of Biden's, 17%
of Trump's). These are almost entirely conversational management ("Go
ahead.", "Yeah, please.") that manufactured cross-turn n-grams like "go ahead
go ahead" and flooded the sentiment sample with neutral one-liners. Longer
turns can still concatenate, so an occasional cross-turn n-gram remains
possible.

## Requirements

Python 3.10+ with `matplotlib` and `nltk`
(`python -m nltk.downloader stopwords punkt_tab vader_lexicon`).
