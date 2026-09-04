import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    },
    "llama3-70b": {
        "name": "Llama-3.1-70B",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "layers_attr": ["model", "layers"],
    },
    "aya-8b": {
        "name": "Aya-Expanse-8B",
        "model_id": "CohereForAI/aya-expanse-8b",
        "layers_attr": ["model", "layers"],
    },
    "aya-32b": {
        "name": "Aya-Expanse-32B",
        "model_id": "CohereForAI/aya-expanse-32b",
        "layers_attr": ["model", "layers"],
    },
}

LANG_CONFIG = {
    "hi": {
        "name": "Hindi",
        "native_file": "flores_hin_Deva.txt",
        "latin_file": "flores_hin_Latn.txt",
    },
    "ar": {
        "name": "Arabic",
        "native_file": "flores_arb_Arab.txt",
        "latin_file": "flores_arb_Latn.txt",
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


# --- Activation collection (mean per layer) ---

class MeanActivationCollector:
    def __init__(self, model_key):
        if model_key not in MODEL_CONFIG:
            raise ValueError(f"Unknown model: {model_key}")
        config = MODEL_CONFIG[model_key]
        self.model_key = model_key
        self.model_name = config["name"]

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
        print(f"Model loaded: {self.n_layers} layers")

    def collect_mean(self, texts):
        """Returns dict layer_idx -> mean activation vector [hidden_dim]."""
        sums = {i: None for i in range(self.n_layers)}
        counts = {i: 0 for i in range(self.n_layers)}
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                # h: [1, seq_len, hidden_dim] -> mean-pool over seq -> [hidden_dim]
                v = h.detach().cpu().float().mean(dim=1).squeeze(0)
                if sums[layer_idx] is None:
                    sums[layer_idx] = v.clone()
                else:
                    sums[layer_idx] += v
                counts[layer_idx] += 1
            return hook

        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        for text in tqdm(texts, desc="Collecting activations"):
            inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                self.model(**inputs)

        for h in hooks:
            h.remove()

        return {i: sums[i] / counts[i] for i in range(self.n_layers)}


# --- Similarity ---

def cosine_per_layer(acts_a, acts_b):
    """Cosine similarity for each layer between two mean activation dicts."""
    sims = []
    for layer in sorted(acts_a.keys()):
        v_a = acts_a[layer].float()
        v_b = acts_b[layer].float()
        sim = torch.nn.functional.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item()
        sims.append(sim)
    return np.array(sims)


# --- Experiment ---

def run_collection(model_key, n_samples, n_runs, base_seed):
    """Collect mean activations for both languages × {native, latin} × n_runs.

    Returns nested dict:
        acts[lang_code][script]  ->  np.ndarray [n_runs, n_layers, hidden_dim]
    Stored as numpy on CPU to keep memory predictable for big models.
    """
    model_config = MODEL_CONFIG[model_key]
    print(f"\n=== Collecting activations with {model_config['name']} ({n_runs} runs) ===\n")

    collector = MeanActivationCollector(model_key)

    # Pre-load text pools
    pools = {}
    for lang_code, lang_cfg in LANG_CONFIG.items():
        pools[lang_code] = {
            "native": load_all_lines(os.path.join(DATA_DIR, lang_cfg["native_file"])),
            "latin": load_all_lines(os.path.join(DATA_DIR, lang_cfg["latin_file"])),
        }
        print(f"{lang_cfg['name']}: {len(pools[lang_code]['native'])} native / "
              f"{len(pools[lang_code]['latin'])} latin lines")

    acts = {lc: {"native": [], "latin": []} for lc in LANG_CONFIG}

    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")
        rng = np.random.default_rng(seed)

        for lang_code in LANG_CONFIG:
            for script in ("native", "latin"):
                texts = sample_texts(pools[lang_code][script], n_samples, rng)
                print(f"  {lang_code}/{script}: collecting...")
                mean_acts = collector.collect_mean(texts)
                # Stack layers -> [n_layers, hidden_dim]
                stacked = torch.stack([mean_acts[i] for i in range(collector.n_layers)]).numpy()
                acts[lang_code][script].append(stacked)

    # -> [n_runs, n_layers, hidden_dim]
    for lc in acts:
        for script in acts[lc]:
            acts[lc][script] = np.stack(acts[lc][script], axis=0)

    return acts, model_config, collector.n_layers


# --- Aggregation ---

def per_run_cosines(acts_a_runs, acts_b_runs):
    """acts_*_runs: [n_runs, n_layers, hidden_dim]. Returns [n_runs, n_layers]."""
    n_runs, n_layers, _ = acts_a_runs.shape
    sims = np.zeros((n_runs, n_layers))
    for r in range(n_runs):
        for L in range(n_layers):
            v_a = torch.from_numpy(acts_a_runs[r, L]).float()
            v_b = torch.from_numpy(acts_b_runs[r, L]).float()
            sims[r, L] = torch.nn.functional.cosine_similarity(
                v_a.unsqueeze(0), v_b.unsqueeze(0)
            ).item()
    return sims


def mean_std(sims):
    return sims.mean(axis=0), sims.std(axis=0)


# --- I/O ---

def save_similarity(acts, model_key, n_samples, n_runs, base_seed):
    """Save raw per-run mean activations + computed similarities per language."""
    for lang_code, lang_cfg in LANG_CONFIG.items():
        out = results_dir(model_key, lang_code)

        nat = acts[lang_code]["native"]
        lat = acts[lang_code]["latin"]
        within_sims = per_run_cosines(nat, lat)

        np.savez(
            os.path.join(out, "cosine_within.npz"),
            sims=within_sims,
            mean=within_sims.mean(axis=0),
            std=within_sims.std(axis=0),
        )

    # Between-language sims (Hindi vs Arabic, same script bucket)
    out = plots_dir(model_key)  # combined plot lives at model level
    lang_codes = list(LANG_CONFIG.keys())
    if len(lang_codes) >= 2:
        a, b = lang_codes[0], lang_codes[1]
        native_sims = per_run_cosines(acts[a]["native"], acts[b]["native"])
        latin_sims = per_run_cosines(acts[a]["latin"], acts[b]["latin"])
        np.savez(
            os.path.join(RESULTS_DIR, model_key, "cosine_between.npz"),
            native_sims=native_sims, latin_sims=latin_sims,
            native_mean=native_sims.mean(axis=0), native_std=native_sims.std(axis=0),
            latin_mean=latin_sims.mean(axis=0), latin_std=latin_sims.std(axis=0),
            lang_a=a, lang_b=b,
        )
    print(f"Similarity results saved under {os.path.join(RESULTS_DIR, model_key)}")


# --- Plotting ---

def _plot_lines_with_band(ax, layers, series, colors, labels):
    """series: list of (mean, std) tuples."""
    for (m, s), c, lbl in zip(series, colors, labels):
        ax.plot(layers, m, label=lbl, color=c, lw=2.5, marker="o", markersize=3, alpha=0.9)
        ax.fill_between(layers, m - s, m + s, color=c, alpha=0.2)


def plot_within_language(acts, model_key, model_config):
    """Per-language cosine sim (native vs latin), plotted together."""
    out = plots_dir(model_key)
    output = os.path.join(out, "within_language.pdf")

    palette = sns.color_palette("tab10")
    series, labels, colors = [], [], []

    for i, (lang_code, lang_cfg) in enumerate(LANG_CONFIG.items()):
        sims = per_run_cosines(acts[lang_code]["native"], acts[lang_code]["latin"])
        series.append(mean_std(sims))
        labels.append(lang_cfg["name"])
        colors.append(palette[i + 1])

    n_layers = series[0][0].shape[0]
    layers = np.arange(n_layers)

    set_icml_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    _plot_lines_with_band(ax, layers, series, colors, labels)

    ax.set_xlabel("Layer Index", fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9, borderpad=0.3, handlelength=1.5)
    ax.tick_params(width=1.0)

    fig.tight_layout(pad=0.3)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Within-language plot saved to {output}")

    # Also write a per-language plot under plots/<model>/<lang>/
    for (m, s), lbl, c, lang_code in zip(series, labels, colors, LANG_CONFIG.keys()):
        lang_out = os.path.join(plots_dir(model_key, lang_code), "within_language.pdf")
        fig, ax = plt.subplots(figsize=(3.25, 2.4))
        ax.plot(layers, m, color=c, lw=2.5, marker="o", markersize=3, alpha=0.9, label=lbl)
        ax.fill_between(layers, m - s, m + s, color=c, alpha=0.2)
        ax.set_xlabel("Layer Index", fontweight="bold")
        ax.set_ylabel("Cosine Similarity", fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left", framealpha=0.9, borderpad=0.3, handlelength=1.5)
        ax.tick_params(width=1.0)
        fig.tight_layout(pad=0.3)
        fig.savefig(lang_out, dpi=300)
        plt.close(fig)
        print(f"Per-language within plot saved to {lang_out}")


def plot_between_language(acts, model_key, model_config):
    """Cross-language cosine sim: native↔native and latin↔latin."""
    lang_codes = list(LANG_CONFIG.keys())
    if len(lang_codes) < 2:
        print("Need >=2 languages for between-language plot, skipping.")
        return

    a, b = lang_codes[0], lang_codes[1]
    native_sims = per_run_cosines(acts[a]["native"], acts[b]["native"])
    latin_sims = per_run_cosines(acts[a]["latin"], acts[b]["latin"])

    out = plots_dir(model_key)
    output = os.path.join(out, "between_language.pdf")

    palette = sns.color_palette("tab10")
    n_layers = native_sims.shape[1]
    layers = np.arange(n_layers)

    set_icml_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    _plot_lines_with_band(
        ax, layers,
        series=[mean_std(native_sims), mean_std(latin_sims)],
        colors=[palette[0], palette[1]],
        labels=["Native↔Native", "Latin↔Latin"],
    )

    ax.set_xlabel("Layer Index", fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9, borderpad=0.3, handlelength=1.5)
    ax.tick_params(width=1.0)

    fig.tight_layout(pad=0.3)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Between-language plot saved to {output}")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Cross-script cosine similarity analysis")
    parser.add_argument("--model", type=str, default="llama3-8b",
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    acts, model_config, n_layers = run_collection(
        args.model, args.n_samples, args.n_runs, args.seed
    )
    save_similarity(acts, args.model, args.n_samples, args.n_runs, args.seed)
    plot_within_language(acts, args.model, model_config)
    plot_between_language(acts, args.model, model_config)


if __name__ == "__main__":
    main()