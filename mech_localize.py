import os
import json
import argparse
import datetime as _dt

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

DATA_DIR    = "dev"
ACT_DIR     = "activations"
ACT_SPLIT   = "dev"
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

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
    "hi": {"name": "Hindi",  "native_file": "flores_hin_Deva.txt", "latin_file": "flores_hin_Latn.txt",
           "native_acts": "flores_hin_Deva.pt", "latin_acts": "flores_hin_Latn.pt",
           "native_unicode_range": (0x0900, 0x097F)},
    "ar": {"name": "Arabic", "native_file": "flores_arb_Arab.txt", "latin_file": "flores_arb_Latn.txt",
           "native_acts": "flores_arb_Arab.pt", "latin_acts": "flores_arb_Latn.pt",
           "native_unicode_range": (0x0600, 0x06FF)},
}

# ────────────────────────────────────────────────────────────────
# Path helpers
# ────────────────────────────────────────────────────────────────

def results_dir(model_key, lang_code, subdir=None):
    p = os.path.join(RESULTS_DIR, model_key, lang_code)
    if subdir:
        p = os.path.join(p, subdir)
    os.makedirs(p, exist_ok=True)
    return p


def plots_dir(model_key, lang_code, subdir=None):
    p = os.path.join(PLOTS_DIR, model_key, lang_code)
    if subdir:
        p = os.path.join(p, subdir)
    os.makedirs(p, exist_ok=True)
    return p


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def get_model_layers(model, layers_attr):
    obj = model
    for attr in layers_attr:
        obj = getattr(obj, attr)
    return obj


def get_attn_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise RuntimeError("No attention submodule found on layer")


def load_texts(path, n=None):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines[:n] if n else lines


def load_steering_vectors(model_key, lang_cfg):
    """v = normalize(latin_mean − native_mean), per layer.

    +v points latin-ward; DLA>0 ⇒ head writes latin-ward; DLA<0 ⇒ native-ward.
    """
    acts_dir = os.path.join(ACT_DIR, model_key, ACT_SPLIT)
    a = torch.load(os.path.join(acts_dir, lang_cfg["native_acts"]))
    b = torch.load(os.path.join(acts_dir, lang_cfg["latin_acts"]))
    vecs = {}
    for k in a:
        d = (b[k] - a[k]).float()
        vecs[k] = d / (torch.norm(d) + 1e-8)
    return vecs


def build_token_script_masks(tokenizer, native_range, vocab_size):
    native_mask = np.zeros(vocab_size, dtype=bool)
    latin_mask  = np.zeros(vocab_size, dtype=bool)
    lo, hi = native_range
    for t in range(vocab_size):
        s = tokenizer.decode([t]).strip()
        if not s:
            continue
        c = s[0]
        cp = ord(c)
        if lo <= cp <= hi:
            native_mask[t] = True
        elif "a" <= c.lower() <= "z":
            latin_mask[t] = True
    return native_mask, latin_mask


def save_json(obj, path):
    def default(o):
        if isinstance(o, np.ndarray):   return o.tolist()
        if isinstance(o, np.integer):   return int(o)
        if isinstance(o, np.floating):  return float(o)
        raise TypeError(f"Unserializable: {type(o)}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


# ────────────────────────────────────────────────────────────────
# Model wrapper
# ────────────────────────────────────────────────────────────────

class ModelWrapper:
    def __init__(self, model_key):
        cfg = MODEL_CONFIG[model_key]
        print(f"Loading {cfg['model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        max_memory = None
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            per_gpu_gb = max(1, int(
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            ) - 10)
            max_memory = {i: f"{per_gpu_gb}GiB" for i in range(n_gpus)}
            max_memory["cpu"] = "0GiB"  # forbid CPU offload
            print(f"  device_map=auto with max_memory={max_memory}")

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
            trust_remote_code=True,
        )
        self.model.eval()
        self.layers    = get_model_layers(self.model, cfg["layers_attr"])
        self.n_layers  = len(self.layers)
        self.n_heads   = self.model.config.num_attention_heads
        self.hidden    = self.model.config.hidden_size
        self.head_dim  = getattr(self.model.config, "head_dim", None) or (self.hidden // self.n_heads)
        self.attn_inner = self.n_heads * self.head_dim
        print(f"  {self.n_layers} layers, {self.n_heads} heads, head_dim={self.head_dim}, "
              f"attn_inner={self.attn_inner}, hidden={self.hidden}")


# ────────────────────────────────────────────────────────────────
# DLA
# ────────────────────────────────────────────────────────────────

def run_dla(wrapper, texts, steering_vecs):
    """Per-(layer, head) mean of <head_contribution_to_residual, v>.

    v points latin-ward, so:
      DLA > 0 ⇒ head writes latin-ward
      DLA < 0 ⇒ head writes native-ward
    """
    n_layers, n_heads = wrapper.n_layers, wrapper.n_heads
    head_dim = wrapper.head_dim

    cache = {}
    handles = []
    for i, layer in enumerate(wrapper.layers):
        attn = get_attn_module(layer)
        def make_hook(idx):
            def pre_hook(mod, inp):
                cache[idx] = inp[0].detach()
            return pre_hook
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(i)))

    # Pre-slice o_proj weights by head
    W_O_heads = {}
    for i, layer in enumerate(wrapper.layers):
        attn = get_attn_module(layer)
        W = attn.o_proj.weight.detach()                     # [hidden, attn_inner]
        W_heads = W.view(wrapper.hidden, n_heads, head_dim) # [hidden, n_heads, head_dim]
        W_O_heads[i] = W_heads.permute(1, 2, 0).contiguous() # [n_heads, head_dim, hidden]

    scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    for text in tqdm(texts, desc="DLA forward"):
        cache.clear()
        inputs = wrapper.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            wrapper.model(**inputs)

        for i in range(n_layers):
            v = steering_vecs[f"layer_{i}"].to(DEVICE).to(torch.bfloat16)  # [hidden]
            x_h = cache[i][0].view(-1, n_heads, head_dim)                  # [T, n_heads, head_dim]
            Wv = torch.einsum("hdk,k->hd", W_O_heads[i].to(torch.bfloat16), v)  # [n_heads, head_dim]
            per_head_per_tok = torch.einsum("thd,hd->th", x_h, Wv)         # [T, n_heads]
            scores[i] += per_head_per_tok.mean(dim=0).float().cpu().numpy()

    for h in handles:
        h.remove()
    scores /= len(texts)
    return scores


# ────────────────────────────────────────────────────────────────
# Patching (bidirectional)
# ────────────────────────────────────────────────────────────────

def cache_last_token_o_proj_input(wrapper, text):
    """Run text, cache per-layer o_proj input at last token: dict i -> [attn_inner]."""
    cache = {}
    handles = []
    for i, layer in enumerate(wrapper.layers):
        attn = get_attn_module(layer)
        def make_hook(idx):
            def pre_hook(mod, inp):
                cache[idx] = inp[0].detach()[0, -1].clone()
            return pre_hook
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(i)))
    inputs = wrapper.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = wrapper.model(**inputs)
    for h in handles:
        h.remove()
    return cache, out.logits[0, -1], inputs


def _script_margin_fn(native_mask, latin_mask):
    n_mask_t = torch.from_numpy(native_mask).to(DEVICE)
    l_mask_t = torch.from_numpy(latin_mask).to(DEVICE)
    def fn(logits):
        nl = logits.masked_fill(~n_mask_t, float("-inf"))
        ll = logits.masked_fill(~l_mask_t, float("-inf"))
        return (torch.logsumexp(nl, dim=-1) - torch.logsumexp(ll, dim=-1)).item()
    return fn


def run_patching_direction(wrapper, src_texts, dst_texts, native_mask, latin_mask,
                           max_pairs, direction_label):
    """For each (src, dst) pair, patch dst's run at last token with src's per-head
    activation. Measure change in script margin (native − latin) at first output token.

    Returns effects[n_layers, n_heads] = mean(m_patched − m_dst_baseline) over pairs.
    """
    n_layers, n_heads = wrapper.n_layers, wrapper.n_heads
    head_dim = wrapper.head_dim
    margin = _script_margin_fn(native_mask, latin_mask)

    effects = np.zeros((n_layers, n_heads), dtype=np.float64)
    src_baselines, dst_baselines = [], []
    n_pairs = min(len(src_texts), len(dst_texts), max_pairs)

    for idx in tqdm(range(n_pairs), desc=f"Patch {direction_label}"):
        src_cache, src_logits, _        = cache_last_token_o_proj_input(wrapper, src_texts[idx])
        dst_cache, dst_logits, dst_inp  = cache_last_token_o_proj_input(wrapper, dst_texts[idx])
        m_src = margin(src_logits); m_dst = margin(dst_logits)
        src_baselines.append(m_src); dst_baselines.append(m_dst)

        for layer_idx in range(n_layers):
            src_heads = src_cache[layer_idx].view(n_heads, head_dim)
            dst_heads = dst_cache[layer_idx].view(n_heads, head_dim)

            for head_idx in range(n_heads):
                patched = dst_heads.clone()
                patched[head_idx] = src_heads[head_idx]
                patched_flat = patched.view(-1)

                attn = get_attn_module(wrapper.layers[layer_idx])
                def pre_hook(mod, inp, _pf=patched_flat):
                    x = inp[0].clone()
                    x[0, -1] = _pf
                    return (x,) + inp[1:]
                handle = attn.o_proj.register_forward_pre_hook(pre_hook)
                with torch.no_grad():
                    out = wrapper.model(**dst_inp)
                handle.remove()
                m_patched = margin(out.logits[0, -1])
                effects[layer_idx, head_idx] += (m_patched - m_dst)

    effects /= n_pairs
    stats = {
        "baseline_src_margin": float(np.mean(src_baselines)),
        "baseline_dst_margin": float(np.mean(dst_baselines)),
        "n_pairs": int(n_pairs),
        "direction": direction_label,
    }
    return effects, stats


# ────────────────────────────────────────────────────────────────
# Top-k selection (signed and absolute variants)
# ────────────────────────────────────────────────────────────────

def top_k(matrix, k, mode="abs"):
    flat = matrix.ravel()
    if mode == "abs":
        idx = np.argsort(-np.abs(flat))[:k]
    elif mode == "pos":
        idx = np.argsort(-flat)[:k]
    elif mode == "neg":
        idx = np.argsort(flat)[:k]
    else:
        raise ValueError(mode)
    out = []
    for i in idx:
        l, h = np.unravel_index(int(i), matrix.shape)
        out.append({"layer": int(l), "head": int(h), "score": float(matrix[l, h])})
    return out


# ────────────────────────────────────────────────────────────────
# Plotting (no titles)
# ────────────────────────────────────────────────────────────────

def set_icml_style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif", "mathtext.fontset": "cm",
    })
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "axes.linewidth": 1.0})


def plot_heatmap(matrix, out_path, center_zero=True):
    set_icml_style()
    n_layers, n_heads = matrix.shape
    w = max(3.25, n_heads * 0.14); h = max(2.4, n_layers * 0.09)
    fig, ax = plt.subplots(figsize=(w, h))
    if center_zero:
        vmax = np.max(np.abs(matrix))
        sns.heatmap(matrix, ax=ax, cmap="RdBu_r", center=0.0,
                    vmin=-vmax, vmax=vmax, cbar_kws={"shrink": 0.8},
                    xticklabels=5, yticklabels=5)
    else:
        sns.heatmap(matrix, ax=ax, cmap="viridis",
                    cbar_kws={"shrink": 0.8}, xticklabels=5, yticklabels=5)
    ax.set_xlabel("Head Index", fontweight="bold")
    ax.set_ylabel("Layer Index", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_comparison_scatter(dla_mat, patch_mat, out_path,
                            dla_label="DLA score", patch_label="Patch effect",
                            highlight_top_k=10):
    set_icml_style()
    dla_flat   = dla_mat.ravel()
    patch_flat = patch_mat.ravel()

    dla_topk   = set(np.argsort(-np.abs(dla_flat))[:highlight_top_k].tolist())
    patch_topk = set(np.argsort(-np.abs(patch_flat))[:highlight_top_k].tolist())
    both = dla_topk & patch_topk
    only_dla   = dla_topk - patch_topk
    only_patch = patch_topk - dla_topk

    colors = np.array(["#cccccc"] * len(dla_flat), dtype=object)
    sizes  = np.full(len(dla_flat), 6.0)
    for i in only_dla:   colors[i] = sns.color_palette("tab10")[0]; sizes[i] = 22
    for i in only_patch: colors[i] = sns.color_palette("tab10")[3]; sizes[i] = 22
    for i in both:       colors[i] = sns.color_palette("tab10")[2]; sizes[i] = 28

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(dla_flat, patch_flat, c=colors.tolist(), s=sizes, alpha=0.9,
               edgecolor="black", linewidth=0.2)
    ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel(dla_label, fontweight="bold")
    ax.set_ylabel(patch_label, fontweight="bold")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([], [], marker="o", linestyle="", color="w", markerfacecolor="#cccccc",
               markersize=5, label="others"),
        Line2D([], [], marker="o", linestyle="", color="w",
               markerfacecolor=sns.color_palette("tab10")[0], markersize=7, label=f"top-{highlight_top_k} DLA"),
        Line2D([], [], marker="o", linestyle="", color="w",
               markerfacecolor=sns.color_palette("tab10")[3], markersize=7, label=f"top-{highlight_top_k} patch"),
        Line2D([], [], marker="o", linestyle="", color="w",
               markerfacecolor=sns.color_palette("tab10")[2], markersize=8, label="both"),
    ]
    ax.legend(handles=legend_elems, loc="best", framealpha=0.9, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_overlap_bar(dla_mat, patch_mat, out_path, ks=(5, 10, 20, 30)):
    set_icml_style()
    dla_abs = np.abs(dla_mat).ravel()
    pat_abs = np.abs(patch_mat).ravel()

    jaccards, sizes = [], []
    for k in ks:
        s_dla = set(np.argsort(-dla_abs)[:k].tolist())
        s_pat = set(np.argsort(-pat_abs)[:k].tolist())
        inter = len(s_dla & s_pat)
        union = len(s_dla | s_pat)
        jaccards.append(inter / union if union else 0.0)
        sizes.append(inter)

    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    x = np.arange(len(ks))
    bars = ax.bar(x, jaccards, color=sns.color_palette("tab10")[2],
                  edgecolor="black", linewidth=0.8)
    for bar, inter, k in zip(bars, sizes, ks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{inter}/{k}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("Jaccard(DLA |·|, patch |·|)", fontweight="bold")
    ax.set_ylim(0, max(jaccards) * 1.25 + 0.05)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300); plt.close(fig)
    print(f"Saved plot: {out_path}")


# ────────────────────────────────────────────────────────────────
# Experiment runners
# ────────────────────────────────────────────────────────────────

def run_dla_experiment(wrapper, args, lang_cfg, model_cfg):
    out_dir  = results_dir(args.model, args.lang, "dla")
    plot_dir = plots_dir(args.model, args.lang, "dla")

    vecs = load_steering_vectors(args.model, lang_cfg)
    native_texts = load_texts(os.path.join(DATA_DIR, lang_cfg["native_file"]), args.n_samples)
    latin_texts  = load_texts(os.path.join(DATA_DIR, lang_cfg["latin_file"]),  args.n_samples)

    print(f"\n[DLA] Computing on {len(native_texts)} native + {len(latin_texts)} latin texts")
    scores_native = run_dla(wrapper, native_texts, vecs)
    scores_latin  = run_dla(wrapper, latin_texts,  vecs)
    diff = scores_native - scores_latin

    np.save(os.path.join(out_dir, "scores_native.npy"), scores_native)
    np.save(os.path.join(out_dir, "scores_latin.npy"),  scores_latin)
    np.save(os.path.join(out_dir, "diff.npy"),          diff)

    save_json({"top_heads_by_abs_diff": top_k(diff, args.top_k, "abs")},
              os.path.join(out_dir, "top.json"))
    save_json({"top_heads_by_native_writing": top_k(diff, args.top_k, "neg")},
              os.path.join(out_dir, "top_native_writers.json"))
    save_json({"top_heads_by_latin_writing":  top_k(diff, args.top_k, "pos")},
              os.path.join(out_dir, "top_latin_writers.json"))

    save_json({
        "model": args.model, "lang": args.lang,
        "n_samples_native": len(native_texts),
        "n_samples_latin":  len(latin_texts),
        "shape": list(diff.shape),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "semantics": {
            "steering_vector": "v = normalize(mean_latin_act - mean_native_act)",
            "dla_positive": "head writes latin-ward",
            "dla_negative": "head writes native-ward",
            "diff_negative": "native-writer (follows input on native text, writes native)",
            "diff_positive": "latin-writer",
        },
    }, os.path.join(out_dir, "meta.json"))

    plot_heatmap(diff,          os.path.join(plot_dir, "diff.pdf"),          center_zero=True)
    plot_heatmap(scores_native, os.path.join(plot_dir, "scores_native.pdf"), center_zero=True)
    plot_heatmap(scores_latin,  os.path.join(plot_dir, "scores_latin.pdf"),  center_zero=True)

    print(f"\n[DLA] Top-10 by |diff|:")
    for t in top_k(diff, 10, "abs"):
        print(f"  L{t['layer']:02d} H{t['head']:02d}  diff={t['score']:+.4f}")
    print(f"[DLA] Top-5 native-writers (most negative diff):")
    for t in top_k(diff, 5, "neg"):
        print(f"  L{t['layer']:02d} H{t['head']:02d}  diff={t['score']:+.4f}")
    print(f"[DLA] Top-5 latin-writers (most positive diff):")
    for t in top_k(diff, 5, "pos"):
        print(f"  L{t['layer']:02d} H{t['head']:02d}  diff={t['score']:+.4f}")

    return diff


def run_patch_experiment(wrapper, args, lang_cfg, model_cfg):
    out_dir  = results_dir(args.model, args.lang, "patch")
    plot_dir = plots_dir(args.model, args.lang, "patch")

    native_texts = load_texts(os.path.join(DATA_DIR, lang_cfg["native_file"]), args.max_patch_pairs)
    latin_texts  = load_texts(os.path.join(DATA_DIR, lang_cfg["latin_file"]),  args.max_patch_pairs)

    vocab_size = wrapper.model.get_output_embeddings().weight.shape[0]
    print(f"\n[PATCH] Building script masks (vocab={vocab_size})")
    native_mask, latin_mask = build_token_script_masks(
        wrapper.tokenizer, lang_cfg["native_unicode_range"], vocab_size,
    )
    print(f"  native-script tokens: {native_mask.sum()}, latin tokens: {latin_mask.sum()}")

    print(f"\n[PATCH] Direction 1/2: native→latin (src=native, dst=latin)")
    effects_n2l, stats_n2l = run_patching_direction(
        wrapper, native_texts, latin_texts, native_mask, latin_mask,
        args.max_patch_pairs, "native→latin",
    )
    print(f"\n[PATCH] Direction 2/2: latin→native (src=latin, dst=native)")
    effects_l2n_raw, stats_l2n = run_patching_direction(
        wrapper, latin_texts, native_texts, native_mask, latin_mask,
        args.max_patch_pairs, "latin→native",
    )
    effects_l2n = -effects_l2n_raw

    effects_symmetric = 0.5 * (np.abs(effects_n2l) + np.abs(effects_l2n))

    np.save(os.path.join(out_dir, "effects_n2l.npy"), effects_n2l)
    np.save(os.path.join(out_dir, "effects_l2n.npy"), effects_l2n)
    np.save(os.path.join(out_dir, "effects_symmetric.npy"), effects_symmetric)
    save_json(stats_n2l, os.path.join(out_dir, "stats_n2l.json"))
    save_json(stats_l2n, os.path.join(out_dir, "stats_l2n.json"))

    save_json({"top_heads_by_abs_effect": top_k(effects_symmetric, args.top_k, "abs")},
              os.path.join(out_dir, "top.json"))
    save_json({"top_heads_n2l_pos": top_k(effects_n2l, args.top_k, "pos"),
               "top_heads_n2l_neg": top_k(effects_n2l, args.top_k, "neg")},
              os.path.join(out_dir, "top_n2l.json"))
    save_json({"top_heads_l2n_pos": top_k(effects_l2n, args.top_k, "pos"),
               "top_heads_l2n_neg": top_k(effects_l2n, args.top_k, "neg")},
              os.path.join(out_dir, "top_l2n.json"))

    save_json({
        "model": args.model, "lang": args.lang,
        "n_pairs": stats_n2l["n_pairs"],
        "shape": list(effects_n2l.shape),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "semantics": {
            "effects_n2l": "src=native, dst=latin. Positive ⇒ head carries native-script info (patch pushed latin output toward native).",
            "effects_l2n": "src=latin, dst=native, sign-flipped. Positive ⇒ head carries latin-script info.",
            "effects_symmetric": "0.5*(|n2l| + |l2n|); ranking of overall script-mediating strength.",
        },
    }, os.path.join(out_dir, "meta.json"))

    plot_heatmap(effects_n2l,       os.path.join(plot_dir, "effects_n2l.pdf"),       center_zero=True)
    plot_heatmap(effects_l2n,       os.path.join(plot_dir, "effects_l2n.pdf"),       center_zero=True)
    plot_heatmap(effects_symmetric, os.path.join(plot_dir, "effects_symmetric.pdf"), center_zero=False)

    print(f"\n[PATCH] Baselines:")
    print(f"  n2l: src={stats_n2l['baseline_src_margin']:+.3f}, dst={stats_n2l['baseline_dst_margin']:+.3f}")
    print(f"  l2n: src={stats_l2n['baseline_src_margin']:+.3f}, dst={stats_l2n['baseline_dst_margin']:+.3f}")
    print(f"[PATCH] Top-10 by symmetric |effect|:")
    for t in top_k(effects_symmetric, 10, "abs"):
        print(f"  L{t['layer']:02d} H{t['head']:02d}  sym={t['score']:.4f}  "
              f"n2l={effects_n2l[t['layer'], t['head']]:+.3f}  "
              f"l2n={effects_l2n[t['layer'], t['head']]:+.3f}")

    return effects_n2l, effects_l2n, effects_symmetric


def run_comparison(args, lang_cfg, model_cfg, dla_diff, eff_n2l, eff_l2n, eff_sym):
    out_dir  = results_dir(args.model, args.lang, "comparison")
    plot_dir = plots_dir(args.model, args.lang, "comparison")

    dla_abs   = np.abs(dla_diff).ravel()
    patch_abs = eff_sym.ravel()
    rho_abs, p_abs = spearmanr(dla_abs, patch_abs)

    patch_signed = eff_n2l - eff_l2n
    rho_signed, p_signed = spearmanr(dla_diff.ravel(), patch_signed.ravel())

    overlaps = {}
    for k in (5, 10, 20, 30, 50):
        s_dla = {(int(x["layer"]), int(x["head"]))
                 for x in top_k(np.abs(dla_diff), k, "abs")}
        s_pat = {(int(x["layer"]), int(x["head"]))
                 for x in top_k(eff_sym, k, "abs")}
        overlaps[f"top_{k}"] = {
            "n_intersection": len(s_dla & s_pat),
            "n_union":        len(s_dla | s_pat),
            "jaccard":        len(s_dla & s_pat) / len(s_dla | s_pat),
            "dla_only":       sorted(list(s_dla - s_pat)),
            "patch_only":     sorted(list(s_pat - s_dla)),
            "shared":         sorted(list(s_dla & s_pat)),
        }

    dir_overlaps = {}
    for k in (5, 10, 20):
        s_dla_n = {(int(x["layer"]), int(x["head"]))
                   for x in top_k(dla_diff, k, "neg")}
        s_pat_n = {(int(x["layer"]), int(x["head"]))
                   for x in top_k(eff_n2l, k, "pos")}
        s_dla_l = {(int(x["layer"]), int(x["head"]))
                   for x in top_k(dla_diff, k, "pos")}
        s_pat_l = {(int(x["layer"]), int(x["head"]))
                   for x in top_k(eff_l2n, k, "pos")}
        dir_overlaps[f"top_{k}"] = {
            "native_writers": {
                "jaccard": len(s_dla_n & s_pat_n) / max(1, len(s_dla_n | s_pat_n)),
                "shared":  sorted(list(s_dla_n & s_pat_n)),
            },
            "latin_writers": {
                "jaccard": len(s_dla_l & s_pat_l) / max(1, len(s_dla_l | s_pat_l)),
                "shared":  sorted(list(s_dla_l & s_pat_l)),
            },
        }

    save_json({
        "model": args.model, "lang": args.lang,
        "spearman": {
            "absolute": {"rho": float(rho_abs), "p_value": float(p_abs),
                         "description": "|DLA diff| vs symmetric |patch effect|"},
            "signed":   {"rho": float(rho_signed), "p_value": float(p_signed),
                         "description": "DLA diff vs (n2l - l2n)."},
        },
        "abs_overlaps": overlaps,
        "direction_specific_overlaps": dir_overlaps,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }, os.path.join(out_dir, "comparison.json"))

    plot_comparison_scatter(
        np.abs(dla_diff), eff_sym,
        os.path.join(plot_dir, "scatter_abs.pdf"),
        dla_label="|DLA diff|", patch_label="|patch effect| (sym)",
        highlight_top_k=10,
    )
    plot_comparison_scatter(
        dla_diff, patch_signed,
        os.path.join(plot_dir, "scatter_signed.pdf"),
        dla_label="DLA diff", patch_label="n2l − l2n",
        highlight_top_k=10,
    )
    plot_overlap_bar(
        dla_diff, eff_sym,
        os.path.join(plot_dir, "overlap_bar.pdf"),
        ks=(5, 10, 20, 30),
    )

    print(f"\n[COMPARE] Spearman (|DLA| vs |patch|): ρ={rho_abs:+.3f}, p={p_abs:.2e}")
    print(f"[COMPARE] Spearman (signed):           ρ={rho_signed:+.3f}, p={p_signed:.2e}")
    print(f"[COMPARE] Overlap Jaccard / intersection:")
    for k_label, d in overlaps.items():
        print(f"  {k_label}: J={d['jaccard']:.2f}  |∩|={d['n_intersection']}  shared={d['shared']}")
    print(f"[COMPARE] Direction-specific Jaccard (top-10):")
    d10 = dir_overlaps["top_10"]
    print(f"  native writers: J={d10['native_writers']['jaccard']:.2f}, shared={d10['native_writers']['shared']}")
    print(f"  latin  writers: J={d10['latin_writers']['jaccard']:.2f}, shared={d10['latin_writers']['shared']}")


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

def load_dla_diff(model_key, lang_code):
    return np.load(os.path.join(results_dir(model_key, lang_code, "dla"), "diff.npy"))


def load_patch_effects(model_key, lang_code):
    out = results_dir(model_key, lang_code, "patch")
    return (np.load(os.path.join(out, "effects_n2l.npy")),
            np.load(os.path.join(out, "effects_l2n.npy")),
            np.load(os.path.join(out, "effects_symmetric.npy")))


def run(args):
    lang_cfg  = LANG_CONFIG[args.lang]
    model_cfg = MODEL_CONFIG[args.model]

    print(f"\n=== {args.experiment.upper()} | {model_cfg['name']} | {lang_cfg['name']} ===")

    needs_model = args.experiment in ("dla", "patch", "all")
    wrapper = ModelWrapper(args.model) if needs_model else None

    dla_diff = None
    eff_n2l = eff_l2n = eff_sym = None

    if args.experiment in ("dla", "all"):
        dla_diff = run_dla_experiment(wrapper, args, lang_cfg, model_cfg)
    if args.experiment in ("patch", "all"):
        eff_n2l, eff_l2n, eff_sym = run_patch_experiment(wrapper, args, lang_cfg, model_cfg)

    if args.experiment in ("compare", "all"):
        if dla_diff is None:
            dla_diff = load_dla_diff(args.model, args.lang)
        if eff_n2l is None:
            eff_n2l, eff_l2n, eff_sym = load_patch_effects(args.model, args.lang)
        run_comparison(args, lang_cfg, model_cfg, dla_diff, eff_n2l, eff_l2n, eff_sym)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_CONFIG, default="llama3-8b")
    p.add_argument("--lang",  choices=LANG_CONFIG,  default="hi")
    p.add_argument("--experiment", choices=["dla", "patch", "compare", "all"], default="all")
    p.add_argument("--n-samples",        type=int, default=200,
                   help="Samples for DLA (forward passes).")
    p.add_argument("--max-patch-pairs",  type=int, default=50,
                   help="Pairs for EACH patching direction (n2l and l2n).")
    p.add_argument("--top-k",            type=int, default=30,
                   help="Top-k to save in JSON ranking files.")
    args = p.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    run(args)


if __name__ == "__main__":
    main()