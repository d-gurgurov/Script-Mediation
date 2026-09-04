import os
import torch
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "dev"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

N_SAMPLES = 1000
N_RUNS = 3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIG = {
    "llama3-8b": {
        "name": "Llama-3.1-8B",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "layers_attr": ["model", "layers"],
        "n_heads": 32,
    },
    "llama3-70b": {
        "name": "Llama-3.1-70B",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "layers_attr": ["model", "layers"],
        "n_heads": 64,
    },
    "aya-8b": {
        "name": "Aya-Expanse-8B",
        "model_id": "CohereForAI/aya-expanse-8b",
        "layers_attr": ["model", "layers"],
        "n_heads": 32,
    },
    "aya-32b": {
        "name": "Aya-Expanse-32B",
        "model_id": "CohereForAI/aya-expanse-32b",
        "layers_attr": ["model", "layers"],
        "n_heads": 64,
    },
}

LANG_CONFIG = {
    "hi": {
        "name": "Hindi",
        "native_file": "flores_hin_Deva.txt",
        "latin_file": "flores_hin_Latn.txt",
        "native_label": "Devanagari",
        "latin_label": "Latin",
    },
    "ar": {
        "name": "Arabic",
        "native_file": "flores_arb_Arab.txt",
        "latin_file": "flores_arb_Latn.txt",
        "native_label": "Arabic",
        "latin_label": "Latin",
    },
}


# --- Path helpers ---

def results_dir(model_key, lang_code):
    p = os.path.join(RESULTS_DIR, model_key, lang_code)
    os.makedirs(p, exist_ok=True)
    return p


def plots_dir(model_key, lang_code=None):
    p = os.path.join(PLOTS_DIR, model_key, lang_code) if lang_code else os.path.join(PLOTS_DIR, model_key)
    os.makedirs(p, exist_ok=True)
    return p


# --- Utilities ---

def get_model_layers(model, layers_attr):
    obj = model
    for attr in layers_attr:
        obj = getattr(obj, attr)
    return obj


def load_all_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sample_texts(lines, n, rng):
    """Sample n texts with replacement if needed, using the provided rng."""
    if n <= len(lines):
        idx = rng.choice(len(lines), size=n, replace=False)
    else:
        idx = rng.choice(len(lines), size=n, replace=True)
    return [lines[i] for i in idx]


def set_icml_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
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


# --- Model & Activation Collection ---

class ActivationCollector:
    def __init__(self, model_key):
        if model_key not in MODEL_CONFIG:
            raise ValueError(f"Unknown model: {model_key}")
        config = MODEL_CONFIG[model_key]
        self.model_key = model_key
        self.model_name = config["name"]
        self.n_heads = config["n_heads"]

        print(f"Loading model: {config['model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.layers = get_model_layers(self.model, config["layers_attr"])
        self.n_layers = len(self.layers)
        self.head_dim = self.model.config.hidden_size // self.n_heads
        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads, head_dim={self.head_dim}")

    def collect(self, texts):
        """Returns:
            layer_acts: dict layer_idx -> [n_samples, hidden_dim]
            head_acts:  dict layer_idx -> [n_samples, n_heads, head_dim]
        """
        layer_acts = {i: [] for i in range(self.n_layers)}
        head_acts = {i: [] for i in range(self.n_layers)}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                h = h.detach().cpu().float()
                h_pooled = h.mean(dim=1).squeeze(0)
                layer_acts[layer_idx].append(h_pooled)
                h_heads = h.squeeze(0).view(-1, self.n_heads, self.head_dim)
                h_heads_pooled = h_heads.mean(dim=0)
                head_acts[layer_idx].append(h_heads_pooled)
            return hook

        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        for text in tqdm(texts, desc="Collecting activations"):
            inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)

        for h in hooks:
            h.remove()

        layer_acts = {i: torch.stack(v).numpy() for i, v in layer_acts.items()}
        head_acts = {i: torch.stack(v).numpy() for i, v in head_acts.items()}
        return layer_acts, head_acts


# --- Probing ---

def make_split(n):
    split = int(0.8 * n)
    train_idx = np.concatenate([np.arange(split), np.arange(n, n + split)])
    val_idx = np.concatenate([np.arange(split, n), np.arange(n + split, 2 * n)])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    return train_idx, val_idx, labels


def _fit_sgd(X_train, y_train, X_val, y_val, seed):
    clf = SGDClassifier(
        loss="log_loss", learning_rate="constant", eta0=1e-4,
        max_iter=1, random_state=seed, alpha=1e-4,
    )
    clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    return accuracy_score(y_val, clf.predict(X_val))


def train_layer_probes(layer_native, layer_latin, n_layers, seed):
    n = layer_native[0].shape[0]
    train_idx, val_idx, labels = make_split(n)
    y_train, y_val = labels[train_idx], labels[val_idx]

    results = []
    for layer in tqdm(range(n_layers), desc="Layer probes"):
        X = np.concatenate([layer_native[layer], layer_latin[layer]], axis=0)
        acc = _fit_sgd(X[train_idx], y_train, X[val_idx], y_val, seed)
        fisher = compute_fisher_ratio(layer_native[layer], layer_latin[layer])
        results.append({"layer": layer, "accuracy": acc, "fisher": fisher})
    return results


def train_head_probes(head_native, head_latin, n_layers, n_heads, seed):
    n = head_native[0].shape[0]
    train_idx, val_idx, labels = make_split(n)
    y_train, y_val = labels[train_idx], labels[val_idx]

    acc_matrix = np.zeros((n_layers, n_heads))
    for layer in tqdm(range(n_layers), desc="Head probes"):
        for head in range(n_heads):
            X = np.concatenate(
                [head_native[layer][:, head, :], head_latin[layer][:, head, :]], axis=0
            )
            acc_matrix[layer, head] = _fit_sgd(X[train_idx], y_train, X[val_idx], y_val, seed)
    return acc_matrix


def compute_fisher_ratio(X_0, X_1):
    X_0 = torch.from_numpy(X_0).float()
    X_1 = torch.from_numpy(X_1).float()
    mu_0, mu_1 = X_0.mean(dim=0), X_1.mean(dim=0)
    n = len(X_0) + len(X_1)
    S_0 = (X_0 - mu_0).T @ (X_0 - mu_0)
    S_1 = (X_1 - mu_1).T @ (X_1 - mu_1)
    S_w = (S_0 + S_1) / n
    reg = 1e-6 * torch.eye(S_w.shape[0])
    mean_diff = mu_1 - mu_0
    fisher = (mean_diff @ torch.linalg.pinv(S_w + reg) @ mean_diff).item()
    return float(fisher)


# --- Experiment ---

def run_probing(model_key, lang_code, n_samples, n_runs, base_seed):
    """Run probing n_runs times with different data subsamples.

    Returns:
        layer_runs: list of n_runs lists of {layer, accuracy, fisher}
        head_runs:  np.ndarray [n_runs, n_layers, n_heads]
    """
    if lang_code not in LANG_CONFIG:
        raise ValueError(f"Unknown language: {lang_code}")

    model_config = MODEL_CONFIG[model_key]
    lang_config = LANG_CONFIG[lang_code]

    print(f"\n=== Probing {lang_config['name']} with {model_config['name']} ({n_runs} runs) ===\n")

    native_lines = load_all_lines(os.path.join(DATA_DIR, lang_config["native_file"]))
    latin_lines = load_all_lines(os.path.join(DATA_DIR, lang_config["latin_file"]))
    print(f"Pool: {len(native_lines)} {lang_config['native_label']} / "
          f"{len(latin_lines)} {lang_config['latin_label']} lines")

    collector = ActivationCollector(model_key)

    layer_runs = []
    head_runs = []

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")
        rng = np.random.default_rng(seed)

        native_texts = sample_texts(native_lines, n_samples, rng)
        latin_texts = sample_texts(latin_lines, n_samples, rng)

        print(f"Collecting {lang_config['native_label']} activations...")
        layer_native, head_native = collector.collect(native_texts)
        print(f"Collecting {lang_config['latin_label']} activations...")
        layer_latin, head_latin = collector.collect(latin_texts)

        layer_results = train_layer_probes(layer_native, layer_latin, collector.n_layers, seed)
        head_acc = train_head_probes(
            head_native, head_latin, collector.n_layers, collector.n_heads, seed
        )

        layer_runs.append(layer_results)
        head_runs.append(head_acc)

        # Free memory before next run
        del layer_native, layer_latin, head_native, head_latin

    head_runs = np.stack(head_runs, axis=0)  # [n_runs, n_layers, n_heads]
    return layer_runs, head_runs, model_config, lang_config


# --- Aggregation ---

def aggregate_layer_runs(layer_runs):
    """layer_runs: list of n_runs lists -> DataFrame with mean/std per layer."""
    rows = []
    for run_idx, results in enumerate(layer_runs):
        for r in results:
            rows.append({"run": run_idx, **r})
    df = pd.DataFrame(rows)
    agg = df.groupby("layer").agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        fish_mean=("fisher", "mean"), fish_std=("fisher", "std"),
    ).reset_index()
    return df, agg


# --- I/O ---

def save_results(layer_runs, head_runs, model_key, lang_code, model_config, lang_config, n_samples, n_runs, base_seed):
    out = results_dir(model_key, lang_code)

    # Per-run layer results (full data)
    df, agg = aggregate_layer_runs(layer_runs)
    layer_path = os.path.join(out, "probe_layer_results.json")
    with open(layer_path, "w") as f:
        json.dump({
            "config": {
                "model": model_key, "model_name": model_config["name"],
                "language": lang_code, "language_name": lang_config["name"],
                "native_label": lang_config["native_label"],
                "latin_label": lang_config["latin_label"],
                "n_samples": n_samples, "n_runs": n_runs, "base_seed": base_seed,
            },
            "runs": [list(r) for r in layer_runs],
            "aggregate": agg.to_dict(orient="records"),
        }, f, indent=4)
    print(f"Layer results saved to {layer_path}")

    # Head accuracy: all runs + mean/std
    head_path = os.path.join(out, "probe_head_acc.npz")
    np.savez(
        head_path,
        runs=head_runs,
        mean=head_runs.mean(axis=0),
        std=head_runs.std(axis=0),
    )
    print(f"Head accuracy saved to {head_path}")


# --- Plotting ---

def plot_layer_accuracy(layer_runs, model_key, lang_code, model_config, lang_config):
    """Per-layer accuracy with std-dev band across runs."""
    out = plots_dir(model_key, lang_code)
    output_plot = os.path.join(out, "probe_accuracy.pdf")

    _, agg = aggregate_layer_runs(layer_runs)
    set_icml_style()

    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    ax.plot(agg["layer"], agg["acc_mean"],
            color="steelblue", lw=2.5, marker="o", markersize=3, label="Accuracy")
    ax.fill_between(agg["layer"],
                    agg["acc_mean"] - agg["acc_std"],
                    agg["acc_mean"] + agg["acc_std"],
                    color="steelblue", alpha=0.2)
    ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.7, label="Chance")
    ax.set_xlabel("Layer Index", fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontweight="bold")
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, borderpad=0.3, handlelength=1.5)
    ax.tick_params(width=1.0)

    fig.tight_layout(pad=0.3)
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)
    print(f"Layer accuracy plot saved to {output_plot}")


def plot_head_heatmap(head_runs, model_key, lang_code, model_config, lang_config):
    """Heatmap of mean head accuracy across runs."""
    out = plots_dir(model_key, lang_code)
    output_plot = os.path.join(out, "probe_head_heatmap.pdf")

    mean_matrix = head_runs.mean(axis=0)
    n_layers, n_heads = mean_matrix.shape
    set_icml_style()

    w = max(3.25, n_heads * 0.14)
    h = max(2.4, n_layers * 0.09)
    fig, ax = plt.subplots(figsize=(w, h))

    sns.heatmap(
        mean_matrix, ax=ax, cmap="RdYlGn",
        vmin=0.5, vmax=1.0, linewidths=0.0,
        cbar_kws={"label": "Accuracy", "shrink": 0.8},
        xticklabels=5, yticklabels=5,
    )
    ax.set_xlabel("Head Index", fontweight="bold")
    ax.set_ylabel("Layer Index", fontweight="bold")
    ax.tick_params(width=1.0, labelsize=8)
    ax.invert_yaxis()

    fig.tight_layout(pad=0.3)
    fig.savefig(output_plot, dpi=300)
    plt.close(fig)
    print(f"Head heatmap saved to {output_plot}")


def plot_combined(all_layer_runs, all_head_runs, model_key, model_config):
    """Aggregate across languages AND runs.

    For each layer: pool accuracy/fisher values from every (lang, run) -> mean ± std.
    For heads: mean over all (lang, run) heatmaps.
    """
    out = plots_dir(model_key)

    # --- Layer plot: accuracy + fisher with std bands ---
    rows = []
    for lang_code, runs in all_layer_runs.items():
        for run_idx, results in enumerate(runs):
            for r in results:
                rows.append({
                    "lang": lang_code, "run": run_idx,
                    "layer": r["layer"], "accuracy": r["accuracy"], "fisher": r["fisher"],
                })
    df = pd.DataFrame(rows)
    agg = df.groupby("layer").agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        fish_mean=("fisher", "mean"), fish_std=("fisher", "std"),
    ).reset_index()

    set_icml_style()
    fig, ax1 = plt.subplots(figsize=(3.25, 2.4))

    ax1.plot(agg["layer"], agg["acc_mean"],
             color="steelblue", lw=2.8, marker="o", markersize=3,
             alpha=0.95, label="Accuracy")
    ax1.fill_between(agg["layer"],
                     agg["acc_mean"] - agg["acc_std"],
                     agg["acc_mean"] + agg["acc_std"],
                     color="steelblue", alpha=0.18)
    ax1.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax1.set_xlabel("Layer Index", fontweight="bold")
    ax1.set_ylabel("Probe Accuracy", fontweight="bold")
    ax1.set_ylim(0.45, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(agg["layer"], agg["fish_mean"],
             color="darkred", lw=2.4, ls="--", marker="s", markersize=2.5,
             alpha=0.9, label="Fisher Ratio")
    ax2.fill_between(agg["layer"],
                     agg["fish_mean"] - agg["fish_std"],
                     agg["fish_mean"] + agg["fish_std"],
                     color="darkred", alpha=0.12)
    ax2.set_ylabel("Fisher Ratio", fontweight="bold")

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2,
               loc="lower right", framealpha=0.9, borderpad=0.3, handlelength=1.5)
    ax1.tick_params(width=1.0)
    ax2.tick_params(width=1.0)

    output_layer = os.path.join(out, "probe_accuracy_combined.pdf")
    fig.tight_layout(pad=0.3)
    fig.savefig(output_layer, dpi=300)
    plt.close(fig)
    print(f"Combined layer plot saved to {output_layer}")

    # --- Head heatmap: average across languages and runs ---
    # all_head_runs[lang] has shape [n_runs, n_layers, n_heads]
    all_stacks = np.concatenate(list(all_head_runs.values()), axis=0)  # [n_langs*n_runs, L, H]
    avg_matrix = all_stacks.mean(axis=0)

    n_layers, n_heads = avg_matrix.shape
    w = max(3.25, n_heads * 0.14)
    h = max(2.4, n_layers * 0.09)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        avg_matrix, ax=ax, cmap="RdYlGn",
        vmin=0.5, vmax=1.0, linewidths=0.0,
        cbar_kws={"label": "Accuracy", "shrink": 0.8},
        xticklabels=5, yticklabels=5,
    )
    ax.set_xlabel("Head Index", fontweight="bold")
    ax.set_ylabel("Layer Index", fontweight="bold")
    ax.tick_params(width=1.0, labelsize=8)
    ax.invert_yaxis()

    output_head = os.path.join(out, "probe_head_heatmap_combined.pdf")
    fig.tight_layout(pad=0.3)
    fig.savefig(output_head, dpi=300)
    plt.close(fig)
    print(f"Combined head heatmap saved to {output_head}")


# --- Runners ---

def run_all_languages(model_key, n_samples, n_runs, base_seed):
    all_layer_runs = {}
    all_head_runs = {}
    model_config = MODEL_CONFIG[model_key]
    for lang_code in LANG_CONFIG:
        layer_runs, head_runs, model_config, lang_config = run_probing(
            model_key, lang_code, n_samples, n_runs, base_seed
        )
        save_results(layer_runs, head_runs, model_key, lang_code,
                     model_config, lang_config, n_samples, n_runs, base_seed)
        plot_layer_accuracy(layer_runs, model_key, lang_code, model_config, lang_config)
        plot_head_heatmap(head_runs, model_key, lang_code, model_config, lang_config)
        all_layer_runs[lang_code] = layer_runs
        all_head_runs[lang_code] = head_runs
    return all_layer_runs, all_head_runs, model_config


def main():
    parser = argparse.ArgumentParser(description="Linear probes for script classification")
    parser.add_argument("--model", type=str, default="llama3-8b",
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--lang", type=str, default=None,
                        choices=list(LANG_CONFIG.keys()),
                        help="Single language (omit for all + combined plots)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.lang:
        layer_runs, head_runs, model_config, lang_config = run_probing(
            args.model, args.lang, args.n_samples, args.n_runs, args.seed
        )
        save_results(layer_runs, head_runs, args.model, args.lang,
                     model_config, lang_config, args.n_samples, args.n_runs, args.seed)
        plot_layer_accuracy(layer_runs, args.model, args.lang, model_config, lang_config)
        plot_head_heatmap(head_runs, args.model, args.lang, model_config, lang_config)
    else:
        all_layer, all_head, model_config = run_all_languages(
            args.model, args.n_samples, args.n_runs, args.seed
        )
        plot_combined(all_layer, all_head, args.model, model_config)


if __name__ == "__main__":
    main()