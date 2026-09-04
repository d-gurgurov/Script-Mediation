import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
FLIP_THRESHOLD = 0.5

# Per-language script + GlotLID full codes
LANG_CONFIG = {
    "hi": {"name": "Hindi",  "native_code": "hin_Deva", "latin_code": "hin_Latn",
           "native_unicode_range": (0x0900, 0x097F)},
    "ar": {"name": "Arabic", "native_code": "arb_Arab", "latin_code": "arb_Latn",
           "native_unicode_range": (0x0600, 0x06FF)},
}

CONDITION_META = {
    "native_input_baseline":            {"target": "native", "group": "baseline", "baseline_for": "native_input"},
    "latin_input_baseline":             {"target": "latin",  "group": "baseline", "baseline_for": "latin_input"},
    "native_input_zero":                {"target": "latin",  "group": "zero",     "baseline_for": "native_input"},
    "latin_input_zero":                 {"target": "native", "group": "zero",     "baseline_for": "latin_input"},
    "native_input_patch_latin_first":   {"target": "latin",  "group": "patch_first", "baseline_for": "native_input"},
    "latin_input_patch_native_first":   {"target": "native", "group": "patch_first", "baseline_for": "latin_input"},
    "native_input_patch_latin_all":     {"target": "latin",  "group": "patch_all",   "baseline_for": "native_input"},
    "latin_input_patch_native_all":     {"target": "native", "group": "patch_all",   "baseline_for": "latin_input"},
}

CONDITION_ORDER = list(CONDITION_META.keys())
INTERVENTION_CONDITIONS = [c for c, m in CONDITION_META.items() if m["group"] != "baseline"]
PATCH_CONDITIONS        = [c for c, m in CONDITION_META.items() if m["group"].startswith("patch")]


# ────────────────────────────────────────────────────────────────
# GlotLID
# ────────────────────────────────────────────────────────────────

class LanguageDetector:
    """Lazy GlotLID wrapper. Initialized once per analyze call."""
    def __init__(self):
        import fasttext
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.model = fasttext.load_model(path)

    @staticmethod
    def _clean(text):
        return " ".join(text.split())

    def detect(self, text):
        """Returns (full_code, script_suffix, confidence) or (empty, empty, 0.0)."""
        text = self._clean(text)
        if not text:
            return "empty", "empty", 0.0
        labels, scores = self.model.predict(text)
        full = labels[0].replace("__label__", "")
        script = full.split("_")[-1] if "_" in full else full
        return full, script, float(scores[0])


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def set_icml_style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif", "mathtext.fontset": "cm",
    })
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "axes.linewidth": 1.0})


def save_json(obj, path):
    def default(o):
        if isinstance(o, np.ndarray):   return o.tolist()
        if isinstance(o, np.integer):   return int(o)
        if isinstance(o, np.floating):  return float(o)
        raise TypeError(f"Unserializable: {type(o)}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def first_char_is_target(text, target_script, native_range):
    for c in text:
        if c.isspace():
            continue
        if target_script == "latin":
            return c.isascii() and c.isalpha()
        lo, hi = native_range
        return lo <= ord(c) <= hi
    return False


# ────────────────────────────────────────────────────────────────
# Per-head-group computation
# ────────────────────────────────────────────────────────────────

def compute_group_metrics(group_entry, lang_cfg, detector):
    """For one head group, compute per-condition char-ratio + GlotLID metrics."""
    prompts = group_entry["prompts"]
    n = len(prompts)
    native_range = lang_cfg["native_unicode_range"]
    target_full_code = {"native": lang_cfg["native_code"], "latin": lang_cfg["latin_code"]}
    target_script_suffix = {"native": lang_cfg["native_code"].split("_")[-1],
                            "latin":  lang_cfg["latin_code"].split("_")[-1]}

    results = {}

    for cond in CONDITION_ORDER:
        meta = CONDITION_META[cond]
        target = meta["target"]
        baseline_cond = f"{meta['baseline_for']}_baseline"
        target_full = target_full_code[target]
        target_script = target_script_suffix[target]

        target_ratios   = []
        native_ratios   = []
        latin_ratios    = []
        flip_successes  = []
        ratio_shifts    = []
        first_char_hits = []
        glotlid_codes   = []
        glotlid_scripts = []
        glotlid_confs   = []
        glotlid_full_hits   = []
        glotlid_script_hits = []

        for p in prompts:
            c = p["conditions"][cond]
            b = p["conditions"][baseline_cond]

            # char-ratio metrics
            t_ratio = c[f"{target}_ratio"]
            b_ratio = b[f"{target}_ratio"]
            target_ratios.append(t_ratio)
            native_ratios.append(c["native_ratio"])
            latin_ratios.append(c["latin_ratio"])
            flip_successes.append(1 if t_ratio >= FLIP_THRESHOLD else 0)
            ratio_shifts.append(t_ratio - b_ratio)
            first_char_hits.append(
                1 if first_char_is_target(c["text"], target, native_range) else 0
            )

            # GlotLID metrics
            full, script, conf = detector.detect(c["text"])
            glotlid_codes.append(full)
            glotlid_scripts.append(script)
            glotlid_confs.append(conf)
            glotlid_full_hits.append(1 if full == target_full else 0)
            glotlid_script_hits.append(1 if script == target_script else 0)

        def _ms(vals):
            arr = np.asarray(vals, dtype=float)
            return (float(arr.mean()),
                    float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)

        m_t, s_t = _ms(target_ratios)
        m_f, s_f = _ms(flip_successes)
        m_gf, s_gf = _ms(glotlid_full_hits)
        m_gs, s_gs = _ms(glotlid_script_hits)

        results[cond] = {
            "target_script":     target,
            "target_full_code":  target_full,

            # raw per-prompt arrays
            "target_ratios":         target_ratios,
            "native_ratios_raw":     native_ratios,
            "latin_ratios_raw":      latin_ratios,
            "flip_successes":        flip_successes,
            "ratio_shifts":          ratio_shifts,
            "first_char_hits":       first_char_hits,
            "glotlid_codes":         glotlid_codes,
            "glotlid_scripts":       glotlid_scripts,
            "glotlid_confidences":   glotlid_confs,
            "glotlid_full_hits":     glotlid_full_hits,
            "glotlid_script_hits":   glotlid_script_hits,

            # aggregates — char-ratio
            "mean_target_ratio":      m_t,  "sem_target_ratio":      s_t,
            "mean_flip_success_rate": m_f,  "sem_flip_success_rate": s_f,
            "mean_ratio_shift":       float(np.mean(ratio_shifts)),
            "mean_first_char_hit":    float(np.mean(first_char_hits)),

            # aggregates — GlotLID
            "mean_glotlid_full_success":   m_gf, "sem_glotlid_full_success":   s_gf,
            "mean_glotlid_script_success": m_gs, "sem_glotlid_script_success": s_gs,
            "mean_glotlid_confidence":     float(np.mean(glotlid_confs)),

            "n": n,
        }

    # Aggregate "patch success" — both metrics
    char_patch_success    = np.mean([results[c]["mean_flip_success_rate"]      for c in PATCH_CONDITIONS])
    glotlid_full_patch    = np.mean([results[c]["mean_glotlid_full_success"]   for c in PATCH_CONDITIONS])
    glotlid_script_patch  = np.mean([results[c]["mean_glotlid_script_success"] for c in PATCH_CONDITIONS])

    # Direction-specific (char-ratio only)
    def _flip(cond):
        return results[cond]["mean_flip_success_rate"]

    n2l_first = _flip("native_input_patch_latin_first")
    n2l_all   = _flip("native_input_patch_latin_all")
    l2n_first = _flip("latin_input_patch_native_first")
    l2n_all   = _flip("latin_input_patch_native_all")
    n2l_mean = (n2l_first + n2l_all) / 2
    l2n_mean = (l2n_first + l2n_all) / 2
    best_dir = "native_to_latin" if n2l_mean >= l2n_mean else "latin_to_native"

    # GlotLID direction-specific (full code)
    def _g(cond):
        return results[cond]["mean_glotlid_full_success"]
    g_n2l_first = _g("native_input_patch_latin_first")
    g_n2l_all   = _g("native_input_patch_latin_all")
    g_l2n_first = _g("latin_input_patch_native_first")
    g_l2n_all   = _g("latin_input_patch_native_all")

    results["_aggregate"] = {
        "mean_patch_flip_success":           float(char_patch_success),
        "mean_patch_glotlid_full_success":   float(glotlid_full_patch),
        "mean_patch_glotlid_script_success": float(glotlid_script_patch),
        "mean_patch_first_first_char": float(np.mean(
            [results[c]["mean_first_char_hit"] for c in PATCH_CONDITIONS if c.endswith("first")])),
        "native_to_latin": {
            "patch_first": float(n2l_first), "patch_all": float(n2l_all), "mean": float(n2l_mean),
            "glotlid_full_first": float(g_n2l_first), "glotlid_full_all": float(g_n2l_all),
        },
        "latin_to_native": {
            "patch_first": float(l2n_first), "patch_all": float(l2n_all), "mean": float(l2n_mean),
            "glotlid_full_first": float(g_l2n_first), "glotlid_full_all": float(g_l2n_all),
        },
        "best_direction":      best_dir,
        "best_direction_mean": float(max(n2l_mean, l2n_mean)),
    }
    return results


# ────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────

def _short_cond_label(cond):
    m = CONDITION_META[cond]
    if m["group"] == "baseline":
        return f"{m['baseline_for'].replace('_input', '')}\nbaseline"
    if m["group"] == "zero":
        return f"{m['baseline_for'].replace('_input', '')}\nzero"
    # patch: name by intervention direction (input script -> injected script)
    inp = m["baseline_for"].replace("_input", "")  # 'native' or 'latin'
    direction = "N2L" if inp == "native" else "L2N"
    dur = "first" if m["group"] == "patch_first" else "all"
    return f"{direction}\n({dur})"


def plot_success_heatmap(group_metrics, group_order, out_path, title,
                         metric_key="mean_flip_success_rate",
                         baseline_metric_key="mean_target_ratio",
                         cbar_label="Flip success rate (intervention)"):
    """Generic heatmap: rows = head groups, cols = conditions."""
    set_icml_style()
    interv_conds = INTERVENTION_CONDITIONS
    base_conds   = [c for c, m in CONDITION_META.items() if m["group"] == "baseline"]

    mat_interv = np.array([
        [group_metrics[g][c][metric_key] for c in interv_conds]
        for g in group_order
    ])
    mat_base = np.array([
        [group_metrics[g][c][baseline_metric_key] for c in base_conds]
        for g in group_order
    ])

    col_labels_interv = [_short_cond_label(c) for c in interv_conds]
    col_labels_base   = [_short_cond_label(c).replace("\nbaseline", "\n(baseline\nsanity)")
                         for c in base_conds]

    w = max(8, 0.85 * (len(interv_conds) + len(base_conds)) + 1.5)
    h = max(3, 0.30 * len(group_order) + 1)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(w, h),
        gridspec_kw={"width_ratios": [len(interv_conds), len(base_conds)], "wspace": 0.08},
    )

    sns.heatmap(mat_interv, ax=ax1, cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=col_labels_interv, yticklabels=group_order,
                cbar_kws={"shrink": 0.8, "label": cbar_label},
                linewidths=0.3, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"size": 7})
    ax1.set_xlabel("Intervention conditions", fontweight="bold")
    ax1.set_ylabel("Head group", fontweight="bold")

    sns.heatmap(mat_base, ax=ax2, cmap="Blues", vmin=0, vmax=1,
                xticklabels=col_labels_base, yticklabels=False,
                cbar_kws={"shrink": 0.8, "label": "Sanity"},
                linewidths=0.3, linecolor="white",
                annot=True, fmt=".2f", annot_kws={"size": 7})
    ax2.set_xlabel("Baseline sanity", fontweight="bold")

    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center", fontsize=7)
    plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=7)
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center", fontsize=7)
    #fig.suptitle(title, fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_baseline_vs_patch(group_metrics, group_order, out_path, title):
    set_icml_style()
    n_groups = len(group_order)
    ncols = min(4, n_groups)
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows), squeeze=False)

    pal = sns.color_palette("tab10")
    for i, group in enumerate(group_order):
        ax = axes[i // ncols, i % ncols]
        gm = group_metrics[group]

        conds = PATCH_CONDITIONS
        x = np.arange(len(conds))
        means = [gm[c]["mean_target_ratio"] for c in conds]
        sems  = [gm[c]["sem_target_ratio"]  for c in conds]
        colors = [pal[0] if c.startswith("native_input") else pal[3] for c in conds]

        ax.bar(x, means, yerr=sems, color=colors, edgecolor="black",
               linewidth=0.6, capsize=3)

        for j, c in enumerate(conds):
            meta = CONDITION_META[c]
            baseline_cond = f"{meta['baseline_for']}_baseline"
            target = meta["target"]
            baseline_ratios = gm[baseline_cond].get(f"{target}_ratios_raw")
            if baseline_ratios is None:
                continue
            b_val = float(np.mean(baseline_ratios))
            ax.plot([j - 0.4, j + 0.4], [b_val, b_val], color="gray",
                    lw=1.3, ls="--", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([_short_cond_label(c) for c in conds], fontsize=6, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.axhline(FLIP_THRESHOLD, color="black", lw=0.6, alpha=0.3)
        #ax.set_title(group, fontsize=8, fontweight="bold")
        if i % ncols == 0:
            ax.set_ylabel("target ratio", fontsize=8)

    for j in range(len(group_order), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    #fig.suptitle(title, fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_prompt_strip(group_metrics, group_order, out_path, title, conditions=None):
    set_icml_style()
    if conditions is None:
        conditions = INTERVENTION_CONDITIONS

    rows = []
    for group in group_order:
        for cond in conditions:
            gm = group_metrics[group][cond]
            for r in gm["target_ratios"]:
                rows.append({"group": group, "condition": _short_cond_label(cond), "target_ratio": r})

    import pandas as pd
    df = pd.DataFrame(rows)

    n_conds = len(conditions)
    w = max(6, 0.9 * n_conds + 2)
    h = max(3, 0.22 * len(group_order) + 2)
    fig, ax = plt.subplots(figsize=(w, h))

    sns.stripplot(
        data=df, x="condition", y="target_ratio", hue="group",
        dodge=True, ax=ax, jitter=0.2, size=3.5, alpha=0.85,
        palette="tab20",
    )
    ax.axhline(FLIP_THRESHOLD, color="black", ls="--", lw=0.7, alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Condition", fontweight="bold")
    ax.set_ylabel("Target-script ratio", fontweight="bold")
    #ax.set_title(title, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, frameon=True)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top_heads_ranked(group_metrics, out_path, title, top_n=20,
                          metric_key="mean_patch_flip_success",
                          xlabel="Mean patch flip-success rate"):
    set_icml_style()
    items = [(g, m["_aggregate"][metric_key]) for g, m in group_metrics.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    items = items[:top_n]

    labels = [g for g, _ in items]
    values = [v for _, v in items]

    def color_for(label):
        if label.startswith("top/"):           return sns.color_palette("tab10")[0]
        if label.startswith("same_layer/"):    return sns.color_palette("tab10")[1]
        if label.startswith("global_bottom/"): return sns.color_palette("tab10")[2]
        if label.startswith("random/"):        return sns.color_palette("tab10")[3]
        return "#888888"

    colors = [color_for(l) for l in labels]

    h = max(2.5, 0.25 * len(labels) + 1)
    fig, ax = plt.subplots(figsize=(5, h))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="black", lw=0.6, alpha=0.3, ls="--")
    ax.set_xlabel(xlabel, fontweight="bold")
    #ax.set_title(title, fontweight="bold")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([], [], marker="s", color="w", markerfacecolor=sns.color_palette("tab10")[0],
               markersize=8, label="top (candidate)"),
        Line2D([], [], marker="s", color="w", markerfacecolor=sns.color_palette("tab10")[1],
               markersize=8, label="same-layer control"),
        Line2D([], [], marker="s", color="w", markerfacecolor=sns.color_palette("tab10")[2],
               markersize=8, label="global-bottom control"),
        Line2D([], [], marker="s", color="w", markerfacecolor=sns.color_palette("tab10")[3],
               markersize=8, label="random control"),
        Line2D([], [], marker="s", color="w", markerfacecolor="#888888",
               markersize=8, label="subset"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=6, framealpha=0.9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ────────────────────────────────────────────────────────────────
# Loading & orchestration
# ────────────────────────────────────────────────────────────────

def load_generations(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["generations"]


def analyze_file(gens_path, out_results_dir, out_plots_dir, lang_cfg, detector, label):
    gens = load_generations(gens_path)

    group_metrics = {}
    for group, entry in gens.items():
        group_metrics[group] = compute_group_metrics(entry, lang_cfg, detector)

    def sort_key(group):
        priority = {"top": 0, "pair": 1, "triple": 1, "top-": 0}
        for prefix, p in priority.items():
            if group.startswith(prefix):
                return (p, group)
        if group.startswith("same_layer"):    return (5, group)
        if group.startswith("global_bottom"): return (6, group)
        if group.startswith("random"):        return (7, group)
        if group.startswith("bottom-"):       return (6, group)
        return (4, group)
    group_order = sorted(group_metrics.keys(), key=sort_key)

    os.makedirs(out_results_dir, exist_ok=True)
    save_json({
        "flip_threshold": FLIP_THRESHOLD,
        "lang_codes": {"native": lang_cfg["native_code"], "latin": lang_cfg["latin_code"]},
        "group_order":    group_order,
        "groups":         group_metrics,
    }, os.path.join(out_results_dir, f"success_metrics_{label}.json"))

    # Rankings — both metrics
    char_ranking = sorted(
        ((g, group_metrics[g]["_aggregate"]["mean_patch_flip_success"])
         for g in group_metrics),
        key=lambda kv: kv[1], reverse=True,
    )
    glotlid_ranking = sorted(
        ((g, group_metrics[g]["_aggregate"]["mean_patch_glotlid_full_success"])
         for g in group_metrics),
        key=lambda kv: kv[1], reverse=True,
    )
    save_json({
        "ranked_char_ratio": [{"group": g, "mean_patch_flip_success": float(v)} for g, v in char_ranking],
        "ranked_glotlid_full": [{"group": g, "mean_patch_glotlid_full_success": float(v)} for g, v in glotlid_ranking],
    }, os.path.join(out_results_dir, f"ranking_{label}.json"))

    # Plots
    os.makedirs(out_plots_dir, exist_ok=True)

    # Char-ratio heatmap
    plot_success_heatmap(
        group_metrics, group_order,
        os.path.join(out_plots_dir, f"success_heatmap_{label}.pdf"),
        f"Char-ratio flip success ({label})",
        metric_key="mean_flip_success_rate",
        baseline_metric_key="mean_target_ratio",
        cbar_label="Flip success rate (char ratio)",
    )

    # GlotLID heatmap
    plot_success_heatmap(
        group_metrics, group_order,
        os.path.join(out_plots_dir, f"success_heatmap_glotlid_{label}.pdf"),
        f"GlotLID full-code success ({label})",
        metric_key="mean_glotlid_full_success",
        baseline_metric_key="mean_glotlid_full_success",
        cbar_label=f"Detected as target lang. code (e.g. {lang_cfg['latin_code']})",
    )

    plot_baseline_vs_patch(
        group_metrics, group_order,
        os.path.join(out_plots_dir, f"baseline_vs_patch_{label}.pdf"),
        f"Target-script ratio under patching vs baseline ({label})",
    )
    plot_per_prompt_strip(
        group_metrics, group_order,
        os.path.join(out_plots_dir, f"per_prompt_strip_{label}.pdf"),
        f"Per-prompt target-script ratios ({label})",
    )
    plot_top_heads_ranked(
        group_metrics,
        os.path.join(out_plots_dir, f"top_heads_ranked_{label}.pdf"),
        f"Head groups by char-ratio patch success ({label})",
        top_n=min(25, len(group_metrics)),
        metric_key="mean_patch_flip_success",
        xlabel="Mean patch flip-success rate (char-ratio)",
    )
    plot_top_heads_ranked(
        group_metrics,
        os.path.join(out_plots_dir, f"top_heads_ranked_glotlid_{label}.pdf"),
        f"Head groups by GlotLID full-code success ({label})",
        top_n=min(25, len(group_metrics)),
        metric_key="mean_patch_glotlid_full_success",
        xlabel="Mean patch GlotLID full-code success",
    )

    # Console summary
    print(f"\n=== Ranking ({label}) — CHAR-RATIO patch flip success ===")
    for g, v in char_ranking[:10]:
        print(f"  {g:<40s}  {v:.3f}")
    print(f"\n=== Ranking ({label}) — GlotLID full-code patch success ===")
    for g, v in glotlid_ranking[:10]:
        print(f"  {g:<40s}  {v:.3f}")


def main():
    p = argparse.ArgumentParser(description="Analyze validation generations.json")
    p.add_argument("--model",  required=True)
    p.add_argument("--lang",   required=True, choices=list(LANG_CONFIG))
    p.add_argument("--source", required=True,
                   help="Head source subdir name (e.g. dla_abs, patch_abs, intersection)")
    p.add_argument("--targets", nargs="+", default=["single", "subsets"],
                   choices=["single", "subsets"])
    args = p.parse_args()

    lang_cfg = LANG_CONFIG[args.lang]

    # Load detector once and reuse across both target files
    print("Loading GlotLID...")
    detector = LanguageDetector()

    base_results = os.path.join(RESULTS_DIR, args.model, args.lang, "validation", args.source)
    base_plots   = os.path.join(PLOTS_DIR,   args.model, args.lang, "validation", args.source)
    out_results  = os.path.join(base_results, "analysis")
    out_plots    = os.path.join(base_plots,   "analysis")

    for tgt in args.targets:
        gens_path = os.path.join(base_results, tgt, "generations.json")
        if not os.path.exists(gens_path):
            print(f"⚠ skipping {tgt}: {gens_path} not found")
            continue
        print(f"\n------ analyzing {tgt} ({gens_path}) ------")
        analyze_file(gens_path, out_results, out_plots, lang_cfg, detector, label=tgt)


if __name__ == "__main__":
    main()