"""
Cassady Shoaff | PresidentSpeech 2026 Revamp
Comparative visuals for the cleaned presidential speech corpora.
Replaces ngrams.py / tokens.py and their per-president toggles.

Methodology changes from the 2021 versions:
  * All frequency charts are per 10,000 tokens, so the two corpora are
    directly comparable despite the ~3x size difference.
  * Lexical diversity uses standardized TTR (mean TTR over 1,000-token
    windows) instead of raw TTR, which shrinks as a corpus grows.
  * New keyness chart: log-likelihood (Dunning's G2) finds the words each
    president uses distinctively more than the other.
  * One stopword list applied to BOTH corpora (the 2021 ngrams run filtered
    'thank'/'please'/'god'/'bless' for Trump only).

Usage:
    python make_visuals.py            # expects ./biden_cleaned.txt etc.
Outputs PNGs to ./visuals/
"""
import math
import pathlib
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

PRESIDENTS = {
    # ColorBrewer RdBu endpoints
    "biden": {"label": "Biden", "color": "#2166ac"},
    "trump": {"label": "Trump", "color": "#b2182b"},
}

OUT_DIR = pathlib.Path("visuals")

# fillers/formulae filtered on top of NLTK's list — applied to both corpora
EXTRA_STOPS = {
    "a.m.", "p.m.", "mr.", "mrs.", "ms.", "dr.",
    "'s", "'re", "'ve", "'ll", "'m", "'d", "n't",
    "yeah", "thank", "thanks", "please", "god", "bless",
}

# Second tier for the "_substantive" chart variants: high-frequency,
# low-information words that crowd topical content out of the n-gram charts.
# Generic evaluatives (great/tremendous/...) are filtered here too — they're
# style, and the standard-stoplist keyness chart already captures them.
# Deliberately NOT filtered: people, country, america(n), new (new york),
# better/best (build back better), first (first lady).
SUBSTANTIVE_EXTRA = {
    # pronouns/aux NLTK misses, contraction fragments
    "us", "let", "would", "could", "should", "must", "may", "might",
    "ca", "wo", "gon", "na", "wan", "ta", "gotta",
    # small numbers
    "one", "two", "three",
    # light verbs
    "get", "gets", "getting", "got", "gotten", "go", "going", "goes",
    "went", "gone", "come", "comes", "coming", "came", "know", "knows",
    "knowing", "knew", "known", "think", "thinks", "thinking", "thought",
    "say", "says", "saying", "said", "see", "sees", "seen", "saw", "look",
    "looks", "looking", "looked", "make", "makes", "making", "made",
    "want", "wants", "wanting", "wanted", "take", "takes", "taking",
    "took", "taken", "put", "give", "gives", "giving", "gave", "tell",
    "tells", "telling", "told", "done", "mean", "means", "meant",
    "happen", "happens", "happened", "happening",
    # discourse markers / degree words
    "well", "really", "actually", "maybe", "also", "even", "still",
    "okay", "ok", "oh", "hey", "sure", "right", "like", "lot", "lots",
    "bit", "little", "much", "many", "way", "ways", "kind", "sort",
    # generic nouns
    "thing", "things", "something", "anything", "everything", "nothing",
    "somebody", "anybody", "everybody", "nobody", "someone", "anyone",
    "everyone",
    # generic time words
    "time", "times", "day", "days", "today", "tonight", "week", "weeks",
    "month", "months", "year", "years", "ago", "never", "ever", "always",
    "every", "back", "ahead",
    # generic evaluatives
    "good", "great", "bad", "nice", "fantastic", "incredible",
    "tremendous", "amazing", "beautiful", "wonderful", "perfect",
    "terrible", "horrible",
}

WINDOW = 1000  # tokens per window for standardized TTR

# Sentences shorter than this many words are dropped before any analysis.
# They are almost entirely conversational management ("Go ahead.", "Yeah.",
# "Thank you."), which manufactures n-grams like "go ahead go ahead" when
# turns are concatenated and floods the sentiment sample with neutrals.
MIN_SENT_WORDS = 4


# ------------------------------------------------------------- tokenize ----

def load_sentences(path):
    """Sentence-split a cleaned corpus and drop ultra-short sentences.

    Returns (kept_sentences, total_sentence_count).
    """
    text = path.read_text(encoding="utf-8")
    # NLTK splits straight apostrophes ("we'll" -> "we", "'ll") but not
    # curly ones, and the corpus uses curly throughout
    text = text.replace("’", "'").replace("‘", "'")
    sentences = nltk.sent_tokenize(text)
    return ([s for s in sentences if len(s.split()) >= MIN_SENT_WORDS],
            len(sentences))


def tokenize(sentences, extra_stops=EXTRA_STOPS):
    stops = set(stopwords.words("english")) | extra_stops
    return [
        w.lower() for w in nltk.word_tokenize(" ".join(sentences))
        if w.lower() not in stops
        and w not in string.punctuation
        and any(c.isalpha() for c in w)
    ]


# ---------------------------------------------------------------- stats ----

def per_10k(counter, total):
    return {w: n / total * 10_000 for w, n in counter.items()}


def lemma_families(tokens, dropped):
    """Collapse inflections AND derivations into one item per word family.

    POS-aware WordNet lemmatization handles irregular forms (said -> say,
    children -> child), then Snowball stemming over the lemma merges
    derivational relatives (vaccine / vaccinated / vaccination -> 'vaccin').
    Stems make poor labels, so each family is labeled with its most frequent
    surface form. Returns Counter {label: count}.
    """
    from nltk.stem import SnowballStemmer, WordNetLemmatizer
    wnl, stemmer = WordNetLemmatizer(), SnowballStemmer("english")
    pos_map = {"J": "a", "V": "v", "N": "n", "R": "r"}
    families, surfaces = Counter(), {}
    for word, tag in nltk.pos_tag(tokens):
        lemma = wnl.lemmatize(word, pos_map.get(tag[0], "n"))
        # a lemma can land in the stoplist even when its surface form
        # doesn't ("saying" -> "say")
        if word in dropped or lemma in dropped:
            continue
        stem = stemmer.stem(lemma)
        families[stem] += 1
        surfaces.setdefault(stem, Counter())[word] += 1
    return Counter({surfaces[stem].most_common(1)[0][0]: count
                    for stem, count in families.items()})


def standardized_ttr(tokens):
    """TTR per non-overlapping 1,000-token window."""
    return [
        len(set(tokens[i:i + WINDOW])) / WINDOW
        for i in range(0, len(tokens) - WINDOW + 1, WINDOW)
    ]


# lemmas that are grammatical machinery rather than lexical verbs:
# auxiliaries/copula, contraction fragments, and politeness formulae
# (modals never reach this filter — they tag as MD, not VB*)
FILLER_VERBS = {
    "be", "have", "do", "thank", "let", "please", "bless",
    "'s", "'re", "'ve", "'m", "'d", "ca", "wo", "sha", "na", "ta",
    "gon", "wan", "got",  # "gotta"/"gonna"/"wanna" fragments
}


def verb_frequencies(sentence_list):
    """Lemma counts of lexical verbs, POS-tagged in sentence context.

    Context matters: future "going to <verb>" and habitual "used to" are
    syntactic filler and are skipped, while genuine "going" (motion) and
    "use" (instrumental) are kept.
    """
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    tagged_sents = nltk.pos_tag_sents(
        nltk.word_tokenize(s) for s in sentence_list)
    counts = Counter()
    for sent in tagged_sents:
        for i, (word, tag) in enumerate(sent):
            if not tag.startswith("VB") or not word.isalpha():
                continue
            word = word.lower()
            nxt = sent[i + 1][0].lower() if i + 1 < len(sent) else ""
            if word in ("going", "used") and nxt == "to":
                continue
            lemma = wnl.lemmatize(word, "v")
            if lemma not in FILLER_VERBS:
                counts[lemma] += 1
    return counts


def log_likelihood(freqs_a, freqs_b, total_a, total_b, min_count=10):
    """Signed Dunning G2 per word: positive = overused in corpus A."""
    scores = {}
    for word in set(freqs_a) | set(freqs_b):
        a, b = freqs_a.get(word, 0), freqs_b.get(word, 0)
        if a + b < min_count:
            continue
        e_a = total_a * (a + b) / (total_a + total_b)
        e_b = total_b * (a + b) / (total_a + total_b)
        g2 = 2 * ((a * math.log(a / e_a) if a else 0)
                  + (b * math.log(b / e_b) if b else 0))
        scores[word] = g2 if a / total_a >= b / total_b else -g2
    return scores


# ----------------------------------------------------------------- plot ----

def style_axis(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)


def paired_barh(data, title, subtitle, fname, n=20):
    """Side-by-side horizontal bar panels, one per president."""
    fig, axes = plt.subplots(1, len(data), figsize=(12, 7))
    for ax, (pres, items) in zip(axes, data.items()):
        cfg = PRESIDENTS[pres]
        labels = [k for k, _ in items][::-1]
        values = [v for _, v in items][::-1]
        ax.barh(labels, values, color=cfg["color"], height=0.72)
        ax.set_title(cfg["label"], fontsize=13, fontweight="bold",
                     color=cfg["color"], pad=10)
        style_axis(ax)
        ax.tick_params(axis="y", labelsize=10)
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8, color="#555555")
        ax.set_xlim(0, max(values) * 1.14)
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.925, subtitle, fontsize=10, color="#666666")
    fig.tight_layout(rect=(0.02, 0, 1, 0.91))
    save(fig, fname)


def keyness_chart(scores, fname="distinctive_words.png", n=15):
    top_a = sorted((s for s in scores.items() if s[1] > 0),
                   key=lambda x: -x[1])[:n]
    top_b = sorted((s for s in scores.items() if s[1] < 0),
                   key=lambda x: x[1])[:n]
    items = top_a[::-1] + top_b              # Biden up top, Trump below
    labels = [w for w, _ in items]
    values = [v for _, v in items]
    colors = [PRESIDENTS["biden" if v > 0 else "trump"]["color"]
              for v in values]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(items)), values, color=colors, height=0.72)
    ax.set_yticks(range(len(items)), labels, fontsize=10)
    ax.axvline(0, color="#999999", linewidth=1)
    style_axis(ax)
    ax.set_xlabel("log-likelihood (G2)  ← Trump   |   Biden →",
                  fontsize=10, color="#666666")
    for i, v in enumerate(values):
        ax.text(v + (8 if v > 0 else -8), i, f"{abs(v):.0f}",
                va="center", ha="left" if v > 0 else "right",
                fontsize=8, color="#555555")
    fig.suptitle("Most distinctive words", fontsize=15, fontweight="bold",
                 x=0.07, ha="left")
    fig.text(0.07, 0.935, "Words each president uses far more than the other "
             "(Dunning log-likelihood, min. 10 occurrences)",
             fontsize=10, color="#666666")
    fig.tight_layout(rect=(0.02, 0, 1, 0.92))
    save(fig, fname)


def sentiment_chart(sentences):
    """Per-sentence VADER polarity: share of pos/neu/neg + score distribution."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = {p: [sia.polarity_scores(s)["compound"] for s in sents]
              for p, sents in sentences.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   width_ratios=(1, 1.3))
    # left: stacked share of positive / neutral / negative sentences
    shades = {"positive": "#4d9971", "neutral": "#cccccc",
              "negative": "#c77b6e"}
    for y, (pres, vals) in enumerate(scores.items()):
        n = len(vals)
        shares = {
            "positive": sum(v >= 0.05 for v in vals) / n,
            "neutral": sum(-0.05 < v < 0.05 for v in vals) / n,
            "negative": sum(v <= -0.05 for v in vals) / n,
        }
        left = 0
        for cls, share in shares.items():
            ax1.barh(y, share, left=left, color=shades[cls], height=0.55)
            if share > 0.06:
                ax1.text(left + share / 2, y, f"{share:.0%}", ha="center",
                         va="center", fontsize=10, color="white",
                         fontweight="bold")
            left += share
    ax1.set_yticks(range(len(scores)),
                   [PRESIDENTS[p]["label"] for p in scores], fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.invert_yaxis()
    style_axis(ax1)
    ax1.xaxis.grid(False)
    ax1.set_xticks([])
    ax1.set_title("share of sentences", fontsize=11, color="#666666")
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c)
                        for c in shades.values()],
               labels=list(shades), loc="upper center", ncols=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

    # right: distribution of compound scores
    import numpy as np
    bins = np.linspace(-1, 1, 41)
    for pres, vals in scores.items():
        cfg = PRESIDENTS[pres]
        ax2.hist(vals, bins=bins, density=True, histtype="step",
                 linewidth=2, color=cfg["color"], label=cfg["label"])
        mean = sum(vals) / len(vals)
        ax2.axvline(mean, color=cfg["color"], linestyle="--", linewidth=1.2)
        ax2.text(mean, ax2.get_ylim()[1] * 0.97, f" {mean:+.3f}",
                 color=cfg["color"], fontsize=9, fontweight="bold",
                 va="top")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.set_xlabel("VADER compound score per sentence "
                   "(− negative ... + positive)", fontsize=10,
                   color="#666666")
    ax2.set_title("distribution (dashed = mean)", fontsize=11,
                  color="#666666")
    ax2.legend(frameon=False, fontsize=10)

    fig.suptitle("Sentence-level sentiment (VADER)", fontsize=15,
                 fontweight="bold", x=0.05, ha="left")
    fig.text(0.05, 0.91, "Computed before stopword filtering; sentences "
             f"under {MIN_SENT_WORDS} words ('Go ahead.', 'Yeah.') are "
             "excluded. VADER rewards evaluative words, so a high score "
             "reads as 'upbeat language', not 'positive policy'.",
             fontsize=10, color="#666666")
    fig.tight_layout(rect=(0.01, 0, 1, 0.88))
    save(fig, "sentiment.png")


def diversity_chart(ttrs):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x, (pres, values) in enumerate(ttrs.items()):
        cfg = PRESIDENTS[pres]
        mean = sum(values) / len(values)
        ax.bar(x, mean, width=0.55, color=cfg["color"], alpha=0.25, zorder=1)
        # one dot per 1,000-token window, jittered
        xs = [x + (i % 9 - 4) * 0.035 for i in range(len(values))]
        ax.scatter(xs, values, s=14, color=cfg["color"], alpha=0.6, zorder=2,
                   edgecolors="none")
        ax.hlines(mean, x - 0.3, x + 0.3, color=cfg["color"], linewidth=2.5,
                  zorder=3)
        ax.text(x, mean + 0.012, f"{mean:.3f}", ha="center", fontsize=11,
                fontweight="bold", color=cfg["color"])
    ax.set_xticks(range(len(ttrs)),
                  [PRESIDENTS[p]["label"] for p in ttrs], fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("type-token ratio per window", fontsize=10, color="#666666")
    fig.suptitle("Lexical diversity (standardized TTR)", fontsize=15,
                 fontweight="bold", x=0.09, ha="left")
    fig.text(0.09, 0.915, f"Each dot is one {WINDOW:,}-token window; "
             "the line is the mean. Higher = more varied vocabulary.",
             fontsize=10, color="#666666")
    fig.tight_layout(rect=(0.02, 0, 1, 0.9))
    save(fig, "lexical_diversity.png")


def save(fig, fname):
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / fname, dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved visuals/{fname}")


# ----------------------------------------------------------------- main ----

VARIANTS = {
    # suffix -> (extra stopword set, subtitle note)
    "": (EXTRA_STOPS, "stopwords removed"),
    "_substantive": (EXTRA_STOPS | SUBSTANTIVE_EXTRA,
                     "extended stoplist: light verbs, discourse markers, "
                     "generic time/evaluative words also removed"),
}


def main():
    sentences, base = {}, {}
    for pres in PRESIDENTS:
        kept, total = load_sentences(pathlib.Path(f"{pres}_cleaned.txt"))
        sentences[pres] = kept
        base[pres] = tokenize(kept)
        print(f"{PRESIDENTS[pres]['label']}: kept {len(kept):,} of {total:,} "
              f"sentences (dropped {total - len(kept):,} under "
              f"{MIN_SENT_WORDS} words)")

    for suffix, (extra, note) in VARIANTS.items():
        dropped = extra - EXTRA_STOPS
        tokens, counts, ngram_counts = {}, {}, {}
        for pres in PRESIDENTS:
            tokens[pres] = [t for t in base[pres] if t not in dropped]
            counts[pres] = Counter(tokens[pres])
            # n-grams always form over the *standard* stream, then keep only
            # those made entirely of surviving words — re-forming them over
            # the filtered stream would join words that were never adjacent
            # ("great job, great job" -> "job job")
            ngram_counts[pres] = {
                n: Counter(" ".join(g) for g in nltk.ngrams(base[pres], n)
                           if not dropped.intersection(g))
                for n in (2, 3)
            }
            print(f"{PRESIDENTS[pres]['label']}{suffix}: "
                  f"{len(tokens[pres]):,} content tokens, "
                  f"{len(counts[pres]):,} types")

        def top_per_10k(counters, n=20):
            return {
                pres: sorted(per_10k(c, len(tokens[pres])).items(),
                             key=lambda x: -x[1])[:n]
                for pres, c in counters.items()
            }

        sub = f"occurrences per 10,000 content tokens ({note})"
        paired_barh(top_per_10k(counts), "Top 20 words", sub,
                    f"top20_words{suffix}.png")
        paired_barh(top_per_10k({p: c[2] for p, c in ngram_counts.items()}),
                    "Top 20 bigrams", sub, f"top_bigrams{suffix}.png")
        paired_barh(top_per_10k({p: c[3] for p, c in ngram_counts.items()}),
                    "Top 20 trigrams", sub, f"top_trigrams{suffix}.png")
        keyness_chart(
            log_likelihood(counts["biden"], counts["trump"],
                           len(tokens["biden"]), len(tokens["trump"])),
            f"distinctive_words{suffix}.png")

        if not suffix:  # diversity uses the standard token stream only
            diversity_chart({p: standardized_ttr(t)
                             for p, t in tokens.items()})

    # word families: substantive stoplist, inflections/derivations merged
    fams = {pres: lemma_families(base[pres], SUBSTANTIVE_EXTRA)
            for pres in PRESIDENTS}
    paired_barh(
        {pres: sorted(per_10k(f, sum(f.values())).items(),
                      key=lambda x: -x[1])[:20]
         for pres, f in fams.items()},
        "Top 20 word families",
        "per 10,000 content tokens; inflections and derivations grouped "
        "(vaccine/vaccinated/vaccination), labeled by most frequent form; "
        "extended stoplist", "top20_lemmas.png")

    # most frequent lexical verbs (POS-tagged in sentence context)
    verbs = {pres: verb_frequencies(sentences[pres]) for pres in PRESIDENTS}
    for pres, v in verbs.items():
        print(f"{PRESIDENTS[pres]['label']}: {sum(v.values()):,} lexical "
              f"verb tokens, {len(v):,} verb lemmas")
    paired_barh(
        {pres: sorted(per_10k(v, sum(v.values())).items(),
                      key=lambda x: -x[1])[:20]
         for pres, v in verbs.items()},
        "Top 20 verbs",
        "per 10,000 lexical verb tokens; inflections merged; auxiliaries, "
        "modals, and filler constructions (future 'going to', habitual "
        "'used to') excluded", "top20_verbs.png")

    # sentiment runs on the same filtered sentences, before stopword removal
    sentiment_chart(sentences)
    print("Done.")


if __name__ == "__main__":
    main()
