import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import fasttext
from huggingface_hub import hf_hub_download

# --- Configuration ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
DIRECTIONS = ["nat2lat", "lat2nat"]

MODEL_CONFIG = {
    "llama3-8b": {"name": "Llama-3.1-8B", "n_layers": 32},
    "llama3-70b": {"name": "Llama-3.1-70B", "n_layers": 80},
    "aya-8b": {"name": "Aya-Expanse-8B", "n_layers": 32},
    "aya-32b": {"name": "Aya-Expanse-32B", "n_layers": 40},
}

LANG_CONFIG = {
    "hi": {"name": "Hindi", "native_code": "hin_Deva", "latin_code": "hin_Latn"},
    "ar": {"name": "Arabic", "native_code": "arb_Arab", "latin_code": "arb_Latn"},
}


# ────────────────────────────────────────────────────────────────
# GlotLID
# ────────────────────────────────────────────────────────────────

class LanguageDetector:
    def __init__(self):
        model_path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.model = fasttext.load_model(model_path)

    @staticmethod
    def _clean(text):
        return " ".join(text.split())

    def detect(self, text):
        text = self._clean(text)
        if not text:
            return ("empty", 0.0)
        labels, scores = self.model.predict(text)
        return (labels[0].replace("__label__", ""), float(scores[0]))


# ────────────────────────────────────────────────────────────────
# Paths & I/O
# ────────────────────────────────────────────────────────────────

def mode_suffix(all_layers):
    return "all_layers" if all_layers else "per_layer"


def input_json_path(model_key, lang_code, direction, all_layers):
    return os.path.join(
        RESULTS_DIR, model_key, lang_code,
        f"steering_{direction}_{mode_suffix(all_layers)}.json",
    )


def output_json_path(model_key, lang_code, direction, all_layers):
    return os.path.join(
        RESULTS_DIR, model_key, lang_code,
        f"steering_{direction}_{mode_suffix(all_layers)}_lid.json",
    )


def plot_path(model_key, lang_code, direction):
    out_dir = os.path.join(PLOTS_DIR, model_key, lang_code)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"steering_{direction}.pdf")


# ────────────────────────────────────────────────────────────────
# Detection pass
# ────────────────────────────────────────────────────────────────

def annotate_with_lid(detector, data, lang_cfg, direction):
    """Add detection fields and a direction-aware success flag."""
    target_code = lang_cfg["latin_code"] if direction == "nat2lat" else lang_cfg["native_code"]

    for category, trials in data.items():
        for trial in trials:
            lang, conf = detector.detect(trial["output_text"])
            trial["detected_lang"] = lang
            trial["lang_confidence"] = conf
            trial["is_target_latin"] = (lang == lang_cfg["latin_code"])
            trial["is_native_script"] = (lang == lang_cfg["native_code"])
            trial["success"] = (lang == target_code)
    return data


def process_direction(detector, model_key, lang_code, direction, all_layers):
    in_path = input_json_path(model_key, lang_code, direction, all_layers)
    if not os.path.exists(in_path):
        print(f"Skipping {direction}: {in_path} not found")
        return None

    print(f"\n=== {direction} | {model_key} | {lang_code} | {mode_suffix(all_layers)} ===")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotate_with_lid(detector, data, LANG_CONFIG[lang_code], direction)

    out_path = output_json_path(model_key, lang_code, direction, all_layers)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved → {out_path}")
    return data


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────

def set_icml_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
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


def _flatten_trials(data, all_layers):
    if all_layers:
        return pd.DataFrame([
            {"scale": t["scale"], "success": float(t["success"])}
            for trials in data.values() for t in trials
        ])
    return pd.DataFrame([
        {"layer": t["layers"][0], "scale": t["scale"], "success": float(t["success"])}
        for trials in data.values() for t in trials
    ])


def plot_success(data, model_key, lang_code, direction, all_layers):
    if data is None:
        return
    set_icml_style()
    df = _flatten_trials(data, all_layers)

    target_label = "Latin" if direction == "nat2lat" else "Native"
    ylabel = f"Switch → {target_label}"

    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    if all_layers:
        sns.barplot(data=df, x="scale", y="success", palette="viridis",
                    errorbar="se", linewidth=1.5, edgecolor="black", ax=ax)
        ax.set_xlabel(r"Scale ($\alpha$)", fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        sns.lineplot(data=df, x="layer", y="success", hue="scale",
                     marker="o", palette="viridis", linewidth=2.5, markersize=4,
                     alpha=0.85, ax=ax)
        ax.set_xlabel("Layer Index", fontweight="bold")
        ax.legend(title=r"$\alpha$", loc="lower right", framealpha=0.9,
                  title_fontsize=9, borderpad=0.3, handlelength=1.5)

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(width=1.0)

    out = plot_path(model_key, lang_code, direction)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Plot saved → {out}")


# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────

def print_summary(data, model_key, lang_code, direction, all_layers):
    if data is None:
        return
    lang_cfg = LANG_CONFIG[lang_code]
    total = sum(len(trials) for trials in data.values())
    success_count = sum(1 for trials in data.values() for t in trials if t["success"])
    latin_count = sum(1 for trials in data.values() for t in trials if t["is_target_latin"])
    native_count = sum(1 for trials in data.values() for t in trials if t["is_native_script"])

    print(f"\n--- Summary: {direction} | {model_key} | {lang_code} | {mode_suffix(all_layers)} ---")
    print(f"Total: {total}")
    print(f"Success ({direction}): {success_count} ({100*success_count/total:.1f}%)")
    print(f"  Detected {lang_cfg['latin_code']}: {latin_count} ({100*latin_count/total:.1f}%)")
    print(f"  Detected {lang_cfg['native_code']}: {native_count} ({100*native_count/total:.1f}%)")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate steering with GlotLID, both directions")
    parser.add_argument("--model", choices=MODEL_CONFIG, default="llama3-8b")
    parser.add_argument("--lang", choices=LANG_CONFIG, default="hi")
    parser.add_argument("--all-layers", action="store_true",
                        help="Evaluate all-layers steering results")
    args = parser.parse_args()

    detector = LanguageDetector()
    for direction in DIRECTIONS:
        data = process_direction(detector, args.model, args.lang, direction, args.all_layers)
        plot_success(data, args.model, args.lang, direction, args.all_layers)
        print_summary(data, args.model, args.lang, direction, args.all_layers)


if __name__ == "__main__":
    main()