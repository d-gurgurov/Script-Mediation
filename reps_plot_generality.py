import os
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import fasttext
from huggingface_hub import hf_hub_download


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

DEFAULT_SCALE = 0.25
DEFAULT_LAYER_TAG = "all"
DEFAULT_COMBINE = "mean"
DEFAULT_SOURCES = "hi+ar"

# Per-direction language subsets.
DIRECTION_GROUPS = {
    "nat2lat": {
        # "source":    ["hi", "ar"],
        "non_latin": ["ru", "zh", "ja", "ko", "el", "th", "fa", "ka", "bo"],
    },
    "lat2nat": {
        "latin": ["de", "es", "fr", "vi", "it", "pt", "tr", "pl", "id"],
    },
}

LANG_LABELS = {
    "hi": "Hindi", "ar": "Arabic", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "el": "Greek", "th": "Thai",
    "fa": "Persian", "ka": "Georgian", "bo": "Tibetan",
    "de": "German", "es": "Spanish", "fr": "French", "vi": "Vietnamese",
    "it": "Italian", "pt": "Portuguese", "tr": "Turkish", "pl": "Polish",
    "id": "Indonesian",
}

GROUP_LABELS = {
    "source": "Sources",
    "non_latin": "Non-Latin",
    "latin": "Latin-script",
}

# Per-language unicode ranges for the char-ratio metric.
# Script labels match GlotLID's ISO 15924 suffixes so legends are consistent
# across the GlotLID and char-ratio plots.
CHAR_SCRIPT_CONFIG = {
    "Latn": {"ranges": None,  # filled in dynamically by ascii-letter check
             "label": "Latn"},
    "Cyrl": {"ranges": [(0x0400, 0x04FF)],                  "label": "Cyrl"},
    "Deva": {"ranges": [(0x0900, 0x097F)],                  "label": "Deva"},
    "Arab": {"ranges": [(0x0600, 0x06FF), (0xFB50, 0xFDFF)],
             "label": "Arab"},
    "Grek": {"ranges": [(0x0370, 0x03FF)],                  "label": "Grek"},
    "Thai": {"ranges": [(0x0E00, 0x0E7F)],                  "label": "Thai"},
    "Hang": {"ranges": [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
             "label": "Hang"},
    "Hani": {"ranges": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
             "label": "Hani"},
    "Jpan": {"ranges": [(0x3040, 0x309F), (0x30A0, 0x30FF)],
             "label": "Jpan"},  # Hiragana + Katakana (Hani is reported separately)
    "Geor": {"ranges": [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
             "label": "Geor"},
    "Tibt": {"ranges": [(0x0F00, 0x0FFF)],                  "label": "Tibt"},
    "Ethi": {"ranges": [(0x1200, 0x137F)],                  "label": "Ethi"},
    "Hebr": {"ranges": [(0x0590, 0x05FF)],                  "label": "Hebr"},
}


# ────────────────────────────────────────────────────────────────
# Script detection — two backends
# ────────────────────────────────────────────────────────────────

class LanguageDetector:
    """GlotLID-based script detection."""
    def __init__(self):
        path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.model = fasttext.load_model(path)

    @staticmethod
    def _clean(text):
        return " ".join(text.split())

    def detect_script(self, text):
        text = self._clean(text)
        if not text:
            return "Empty"
        labels, _ = self.model.predict(text)
        full = labels[0].replace("__label__", "")
        return full.split("_")[-1] if "_" in full else full


def detect_script_charratio(text):
    """Return the dominant script of `text` by character count, using unicode
    ranges from CHAR_SCRIPT_CONFIG. Ties broken by config order. Empty/
    whitespace-only text returns 'Empty'."""
    text = "".join(c for c in text if not c.isspace())
    if not text:
        return "Empty"

    counts = {}
    for c in text:
        cp = ord(c)
        if "a" <= c.lower() <= "z":
            counts["Latn"] = counts.get("Latn", 0) + 1
            continue
        for script, cfg in CHAR_SCRIPT_CONFIG.items():
            if script == "Latn":
                continue
            ranges = cfg["ranges"]
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[script] = counts.get(script, 0) + 1
                break

    if not counts:
        return "Other"
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ────────────────────────────────────────────────────────────────
# I/O
# ────────────────────────────────────────────────────────────────

def generality_path(model_key, sources, combine, layer_tag, scale, direction):
    return os.path.join(
        RESULTS_DIR, model_key, "generality",
        f"{sources}_{combine}_{layer_tag}_S{scale}_{direction}.json",
    )


def load_generality(path):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────
# Per-language script distribution
# ────────────────────────────────────────────────────────────────

def detect_distribution(detect_fn, data, languages):
    """Generic distribution builder. `detect_fn` maps a text string to a script label."""
    rows = []
    for lang in languages:
        if lang not in data["results"]:
            print(f"  {lang}: not in JSON, skipping")
            continue
        outputs = [item["steered"] for item in data["results"][lang]]
        if not outputs:
            continue
        scripts = [detect_fn(o) for o in outputs]
        counts = Counter(scripts)
        total = sum(counts.values())
        for script, c in counts.items():
            rows.append({"lang": lang, "script": script, "fraction": c / total})
    return pd.DataFrame(rows)


def keep_global_top_k(df, k=7):
    """Across all languages, keep the k most prevalent scripts; pool rest as 'Other'."""
    totals = df.groupby("script")["fraction"].sum().sort_values(ascending=False)
    keep = set(totals.head(k).index.tolist())

    rows = []
    for lang, sub in df.groupby("lang"):
        kept = sub[sub["script"].isin(keep)]
        rest_frac = sub[~sub["script"].isin(keep)]["fraction"].sum()
        rows.append(kept)
        if rest_frac > 0:
            rows.append(pd.DataFrame([{"lang": lang, "script": "Other",
                                       "fraction": rest_frac}]))
    return pd.concat(rows, ignore_index=True)


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────

def set_icml_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    })
    sns.set_theme(style="whitegrid", rc={
        "grid.linewidth": 0.5,
        "axes.linewidth": 1.0,
    })


def ordered_languages(direction):
    """Flat list of (lang, group) in display order for a given direction."""
    out = []
    for group, langs in DIRECTION_GROUPS[direction].items():
        for l in langs:
            out.append((l, group))
    return out


def plot_stacked(df, model_key, direction, out_path):
    """Stacked bar: x=language, y=fraction, color=script."""
    set_icml_style()

    lang_order = [l for l, _ in ordered_languages(direction) if l in df["lang"].unique()]
    pivot = (df.pivot_table(index="lang", columns="script",
                            values="fraction", fill_value=0.0)
               .reindex(lang_order))

    col_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    if "Other" in col_order:
        col_order.remove("Other")
        col_order.append("Other")
    pivot = pivot[col_order]

    base_palette = sns.color_palette("tab20", n_colors=20)
    non_other = [s for s in col_order if s != "Other"]
    color_map = {script: base_palette[i % len(base_palette)]
                 for i, script in enumerate(non_other)}
    if "Other" in col_order:
        color_map["Other"] = (0.7, 0.7, 0.7)

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for script in col_order:
        vals = pivot[script].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=script,
               color=color_map[script], edgecolor="white", linewidth=0.4, width=0.85)
        bottom += vals

    xtick_labels = [LANG_LABELS.get(l, l) for l in pivot.index]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, rotation=35, ha="right")

    group_order = [l for l, _ in ordered_languages(direction) if l in pivot.index]
    group_of = {l: g for l, g in ordered_languages(direction)}
    boundaries = []
    last_group = None
    for i, l in enumerate(group_order):
        g = group_of[l]
        if last_group is not None and g != last_group:
            boundaries.append(i - 0.5)
        last_group = g
    for b in boundaries:
        ax.axvline(b, color="black", lw=0.6, alpha=0.4, ymin=0, ymax=1)

    ax.set_ylabel("Output Script", fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              framealpha=0.9, borderpad=0.3, handlelength=1.2,
              title="Detected\nScript", title_fontsize=8)
    ax.tick_params(width=1.0)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ────────────────────────────────────────────────────────────────
# Auto metric
# ────────────────────────────────────────────────────────────────

def script_flip_rate(detect_fn, data, languages, direction):
    """Fraction of (lang, prompt) pairs where steered script differs from baseline."""
    per_group = {}
    group_of = {l: g for l, g in ordered_languages(direction)}
    for lang in languages:
        if lang not in data["results"]:
            continue
        items = data["results"][lang]
        if not items:
            continue
        flips = 0
        for it in items:
            b = detect_fn(it["baseline"])
            s = detect_fn(it["steered"])
            if b != s:
                flips += 1
        rate = flips / len(items)
        g = group_of.get(lang, "other")
        per_group.setdefault(g, []).append((lang, rate))
    return per_group


def print_flip_summary(per_group, direction, metric_name):
    print(f"\n--- script-flip rate ({direction}, {metric_name}) ---")
    for g, rows in per_group.items():
        if not rows:
            continue
        mean = np.mean([r for _, r in rows])
        print(f"  {GROUP_LABELS.get(g, g)}: mean={mean:.2f}  "
              + ", ".join(f"{l}={r:.2f}" for l, r in rows))


# ────────────────────────────────────────────────────────────────
# Per-metric pipeline
# ────────────────────────────────────────────────────────────────

def run_pass(model_key, data, direction, languages, detect_fn,
             out_path, top_k, metric_name):
    df = detect_distribution(detect_fn, data, languages)
    df = keep_global_top_k(df, k=top_k)
    plot_stacked(df, model_key, direction, out_path)

    per_group = script_flip_rate(detect_fn, data, languages, direction)
    print_flip_summary(per_group, direction, metric_name)


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot generality of steering directions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sources", default=DEFAULT_SOURCES,
                        help="Sources tag in filename (e.g. 'hi+ar')")
    parser.add_argument("--combine", default=DEFAULT_COMBINE)
    parser.add_argument("--layer-tag", default=DEFAULT_LAYER_TAG,
                        help="'all' for all-layers, or 'L<n>' for single layer")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--top-k", type=int, default=7)
    args = parser.parse_args()

    print("Loading GlotLID...")
    detector = LanguageDetector()
    glotlid_fn = detector.detect_script

    out_dir = os.path.join(PLOTS_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)

    for direction in ("nat2lat", "lat2nat"):
        languages = [l for l, _ in ordered_languages(direction)]
        path = generality_path(args.model, args.sources, args.combine,
                               args.layer_tag, args.scale, direction)
        data = load_generality(path)
        if data is None:
            continue

        # Pass 1: GlotLID
        run_pass(
            args.model, data, direction, languages, glotlid_fn,
            os.path.join(out_dir, f"generality_{direction}.pdf"),
            args.top_k, "GlotLID",
        )
        # Pass 2: char-ratio
        run_pass(
            args.model, data, direction, languages, detect_script_charratio,
            os.path.join(out_dir, f"generality_{direction}_charratio.pdf"),
            args.top_k, "char-ratio",
        )


if __name__ == "__main__":
    main()