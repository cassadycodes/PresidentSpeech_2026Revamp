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

WINDOW = 1000  # tokens per window for standardized TTR


# ------------------------------------------------------------- tokenize ----

def tokenize(path):
    text = path.read_text(encoding="utf-8")
    # NLTK splits straight apostrophes ("we'll" -> "we", "'ll") but not
    # curly ones, and the corpus uses curly throughout
    text = text.replace("’", "'").replace("‘", "'")
    stops = set(stopwords.words("english")) | EXTRA_STOPS
    return [
        w.lower() for w in nltk.word_tokenize(text)
        if w.lower() not in stops
        and w not in string.punctuation
        and any(c.isalpha() for c in w)
    ]


# ---------------------------------------------------------------- stats ----

def per_10k(counter, total):
    return {w: n / total * 10_000 for w, n in counter.items()}


def standardized_ttr(tokens):
    """TTR per non-overlapping 1,000-token window."""
    return [
        len(set(tokens[i:i + WINDOW])) / WINDOW
        for i in range(0, len(tokens) - WINDOW + 1, WINDOW)
    ]


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


def keyness_chart(scores, n=15):
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
    save(fig, "distinctive_words.png")


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

def main():
    tokens, counts, ngram_counts = {}, {}, {}
    for pres in PRESIDENTS:
        tokens[pres] = tokenize(pathlib.Path(f"{pres}_cleaned.txt"))
        counts[pres] = Counter(tokens[pres])
        ngram_counts[pres] = {
            2: Counter(" ".join(g) for g in nltk.bigrams(tokens[pres])),
            3: Counter(" ".join(g) for g in nltk.trigrams(tokens[pres])),
        }
        print(f"{PRESIDENTS[pres]['label']}: {len(tokens[pres]):,} content "
              f"tokens, {len(counts[pres]):,} types")

    def top_per_10k(counters, n=20):
        return {
            pres: sorted(per_10k(c, len(tokens[pres])).items(),
                         key=lambda x: -x[1])[:n]
            for pres, c in counters.items()
        }

    print("Rendering charts...")
    sub = "occurrences per 10,000 content tokens (stopwords removed)"
    paired_barh(top_per_10k(counts), "Top 20 words", sub, "top20_words.png")
    paired_barh(top_per_10k({p: c[2] for p, c in ngram_counts.items()}),
                "Top 20 bigrams", sub, "top_bigrams.png")
    paired_barh(top_per_10k({p: c[3] for p, c in ngram_counts.items()}),
                "Top 20 trigrams", sub, "top_trigrams.png")

    keyness_chart(log_likelihood(counts["biden"], counts["trump"],
                                 len(tokens["biden"]), len(tokens["trump"])))
    diversity_chart({p: standardized_ttr(t) for p, t in tokens.items()})
    print("Done.")


if __name__ == "__main__":
    main()
