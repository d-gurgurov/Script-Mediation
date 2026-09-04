import os
import json
import math
import argparse
from itertools import combinations

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
DATA_DIR    = "data"
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
    "hi": {"name": "Hindi",  "native_unicode_range": (0x0900, 0x097F),
           "native_file": "hi.txt",     "latin_file": "hi_rom.txt"},
    "ar": {"name": "Arabic", "native_unicode_range": (0x0600, 0x06FF),
           "native_file": "ar.txt",     "latin_file": "ar_rom.txt"},
}


def load_validation_prompts(lang_code, n_prompts):
    """Load parallel native/latin prompt pairs from data/<file>.txt."""
    cfg = LANG_CONFIG[lang_code]
    native_path = os.path.join(DATA_DIR, cfg["native_file"])
    latin_path  = os.path.join(DATA_DIR, cfg["latin_file"])
    with open(native_path, encoding="utf-8") as f:
        native = [l.strip() for l in f if l.strip()]
    with open(latin_path, encoding="utf-8") as f:
        latin = [l.strip() for l in f if l.strip()]
    if len(native) != len(latin):
        raise ValueError(
            f"Prompt files not parallel: {len(native)} native vs {len(latin)} latin "
            f"({native_path}, {latin_path})"
        )
    n = min(n_prompts, len(native))
    return {"native": native[:n], "latin": latin[:n]}

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
    for a in layers_attr:
        obj = getattr(obj, a)
    return obj


def get_attn_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise RuntimeError("No attention submodule found on layer")


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


def script_margin_fn(native_mask, latin_mask):
    n_mask_cpu = torch.from_numpy(native_mask)
    l_mask_cpu = torch.from_numpy(latin_mask)
    log_n = math.log(int(native_mask.sum()))
    log_l = math.log(int(latin_mask.sum()))
    cache = {"device": None, "n": None, "l": None}
    def fn(logits):
        if cache["device"] != logits.device:
            cache["device"] = logits.device
            cache["n"] = n_mask_cpu.to(logits.device)
            cache["l"] = l_mask_cpu.to(logits.device)
        nl = logits.masked_fill(~cache["n"], float("-inf"))
        ll = logits.masked_fill(~cache["l"], float("-inf"))
        return (torch.logsumexp(nl, dim=-1) - log_n
                - torch.logsumexp(ll, dim=-1) + log_l).item()
    return fn


def save_json(obj, path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Unserializable: {type(o)}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def set_icml_style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif", "mathtext.fontset": "cm",
    })
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "axes.linewidth": 1.0})


# ────────────────────────────────────────────────────────────────
# Model wrapper
# ────────────────────────────────────────────────────────────────

class ModelWrapper:
    def __init__(self, model_key, eager_attn=True):
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

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )
        if eager_attn:
            load_kwargs["attn_implementation"] = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **load_kwargs)
        self.model.eval()
        self.layers    = get_model_layers(self.model, cfg["layers_attr"])
        self.n_layers  = len(self.layers)
        self.n_heads   = self.model.config.num_attention_heads
        self.hidden    = self.model.config.hidden_size
        self.head_dim  = getattr(self.model.config, "head_dim", None) or (self.hidden // self.n_heads)
        self.attn_inner = self.n_heads * self.head_dim
        print(f"  {self.n_layers} layers, {self.n_heads} heads, head_dim={self.head_dim}, "
              f"attn_inner={self.attn_inner}")


# ────────────────────────────────────────────────────────────────
# Core: cache all layers' last-token o_proj inputs in one forward pass
# ────────────────────────────────────────────────────────────────

def cache_last_token_activations(wrapper, text):
    cache = {}
    handles = []
    for i, layer in enumerate(wrapper.layers):
        attn = get_attn_module(layer)
        def make_hook(idx):
            def pre_hook(mod, inp):
                cache[idx] = inp[0].detach()[0, -1].clone().view(
                    wrapper.n_heads, wrapper.head_dim
                )
            return pre_hook
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(i)))

    inputs = wrapper.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = wrapper.model(**inputs)
    for h in handles:
        h.remove()
    return cache, out.logits[0, -1]


def run_with_intervention(wrapper, text, interventions):
    handles = []
    for layer_idx, head_ops in interventions.items():
        attn = get_attn_module(wrapper.layers[layer_idx])
        def make_hook(ops):
            def pre_hook(mod, inp):
                x = inp[0].clone()
                x_h = x.view(x.shape[0], x.shape[1], wrapper.n_heads, wrapper.head_dim)
                for head_idx, repl in ops:
                    if repl is None:
                        x_h[:, -1, head_idx, :] = 0.0
                    else:
                        x_h[:, -1, head_idx, :] = repl.to(x_h.dtype).to(x_h.device)
                return (x_h.view(x.shape[0], x.shape[1], -1),) + inp[1:]
            return pre_hook
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(head_ops)))

    inputs = wrapper.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = wrapper.model(**inputs)

    for h in handles:
        h.remove()
    return out.logits[0, -1]


# ────────────────────────────────────────────────────────────────
# Generation under intervention
# ────────────────────────────────────────────────────────────────

def _format_prompt(wrapper, user_msg):
    if hasattr(wrapper.tokenizer, "apply_chat_template"):
        try:
            return wrapper.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return user_msg


def _install_group_hook(wrapper, interventions, fire_once=False):
    state = {"fired": False}
    handles = []
    for layer_idx, head_ops in interventions.items():
        attn = get_attn_module(wrapper.layers[layer_idx])
        def make_hook(ops):
            def pre_hook(mod, inp):
                if fire_once and state["fired"]:
                    return None
                x = inp[0].clone()
                x_h = x.view(x.shape[0], x.shape[1], wrapper.n_heads, wrapper.head_dim)
                for head_idx, repl in ops:
                    if repl is None:
                        x_h[:, -1, head_idx, :] = 0.0
                    else:
                        x_h[:, -1, head_idx, :] = repl.to(x_h.dtype).to(x_h.device)
                return (x_h.view(x.shape[0], x.shape[1], -1),) + inp[1:]
            return pre_hook
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(head_ops)))

    if fire_once:
        last_attn = get_attn_module(wrapper.layers[wrapper.n_layers - 1])
        def fire_flag_hook(mod, inp, out):
            state["fired"] = True
        handles.append(last_attn.o_proj.register_forward_hook(fire_flag_hook))

    return handles


def _free_generate(wrapper, text, max_new_tokens):
    formatted = _format_prompt(wrapper, text)
    inputs = wrapper.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        gen = wrapper.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=wrapper.tokenizer.pad_token_id,
        )
    return wrapper.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True).strip()


def _gen_with_interventions(wrapper, text, interventions, max_new_tokens, fire_once):
    handles = _install_group_hook(wrapper, interventions, fire_once=fire_once)
    try:
        return _free_generate(wrapper, text, max_new_tokens)
    finally:
        for h in handles:
            h.remove()


def _char_script_ratio(text, native_range):
    lo, hi = native_range
    lat = nat = 0
    for c in text:
        cp = ord(c)
        if "a" <= c.lower() <= "z":
            lat += 1
        elif lo <= cp <= hi:
            nat += 1
    total = lat + nat
    return (lat / total if total else 0.0, nat / total if total else 0.0)


def generate_for_group(wrapper, head_group, native_texts, latin_texts,
                       native_caches, latin_caches, native_range,
                       max_new_tokens, baseline_gens=None):
    per_layer = {}
    for l, h in head_group:
        per_layer.setdefault(l, []).append(h)

    out = {"prompts": []}
    n_pairs = min(len(native_texts), len(latin_texts))
    for i in tqdm(range(n_pairs), desc="  generating", leave=False):
        entry = {"native_prompt": native_texts[i], "latin_prompt": latin_texts[i],
                 "conditions": {}}

        zero_interv = {l: [(h, None) for h in hs] for l, hs in per_layer.items()}
        patch_NP = {l: [(h, latin_caches[i]["act"][l][h]) for h in hs]
                    for l, hs in per_layer.items()}
        patch_LP = {l: [(h, native_caches[i]["act"][l][h]) for h in hs]
                    for l, hs in per_layer.items()}

        if baseline_gens is not None:
            bn = baseline_gens["native"][i]
            bl = baseline_gens["latin"][i]
        else:
            bn = _free_generate(wrapper, native_texts[i], max_new_tokens)
            bl = _free_generate(wrapper, latin_texts[i], max_new_tokens)

        z_nat = _gen_with_interventions(wrapper, native_texts[i], zero_interv, max_new_tokens, fire_once=False)
        z_lat = _gen_with_interventions(wrapper, latin_texts[i], zero_interv, max_new_tokens, fire_once=False)

        pl_nat = _gen_with_interventions(wrapper, native_texts[i], patch_NP, max_new_tokens, fire_once=True)
        pl_lat = _gen_with_interventions(wrapper, latin_texts[i], patch_LP, max_new_tokens, fire_once=True)

        pa_nat = _gen_with_interventions(wrapper, native_texts[i], patch_NP, max_new_tokens, fire_once=False)
        pa_lat = _gen_with_interventions(wrapper, latin_texts[i], patch_LP, max_new_tokens, fire_once=False)

        for name, txt in [
            ("native_input_baseline",          bn),
            ("latin_input_baseline",           bl),
            ("native_input_zero",              z_nat),
            ("latin_input_zero",               z_lat),
            ("native_input_patch_latin_first", pl_nat),
            ("latin_input_patch_native_first", pl_lat),
            ("native_input_patch_latin_all",   pa_nat),
            ("latin_input_patch_native_all",   pa_lat),
        ]:
            lr, nr = _char_script_ratio(txt, native_range)
            entry["conditions"][name] = {"text": txt, "latin_ratio": lr, "native_ratio": nr}

        out["prompts"].append(entry)

    out["summary"] = {}
    for cond in [
        "native_input_baseline",          "latin_input_baseline",
        "native_input_zero",              "latin_input_zero",
        "native_input_patch_latin_first", "latin_input_patch_native_first",
        "native_input_patch_latin_all",   "latin_input_patch_native_all",
    ]:
        lrs = [p["conditions"][cond]["latin_ratio"]  for p in out["prompts"]]
        nrs = [p["conditions"][cond]["native_ratio"] for p in out["prompts"]]
        out["summary"][cond] = {
            "latin_ratio_mean":  float(np.mean(lrs)),
            "native_ratio_mean": float(np.mean(nrs)),
            "n": len(lrs),
        }
    return out


def precompute_baseline_generations(wrapper, native_texts, latin_texts, max_new_tokens):
    out = {"native": [], "latin": []}
    for t in tqdm(native_texts, desc="Baseline gens (native)", leave=False):
        out["native"].append(_free_generate(wrapper, t, max_new_tokens))
    for t in tqdm(latin_texts, desc="Baseline gens (latin)", leave=False):
        out["latin"].append(_free_generate(wrapper, t, max_new_tokens))
    return out


# ────────────────────────────────────────────────────────────────
# The four-cell measurement, per head-group
# ────────────────────────────────────────────────────────────────

def measure_head_group(wrapper, head_group, native_texts, latin_texts,
                       native_caches, latin_caches, margin_fn):
    per_layer = {}
    for l, h in head_group:
        per_layer.setdefault(l, []).append(h)

    n0, np_, l0, lp = [], [], [], []
    bN, bL = [], []

    n_pairs = min(len(native_texts), len(latin_texts))
    for i in range(n_pairs):
        nat_logits = native_caches[i]["logits"]
        lat_logits = latin_caches[i]["logits"]
        bN.append(margin_fn(nat_logits))
        bL.append(margin_fn(lat_logits))

        zero_interv = {l: [(h, None) for h in hs] for l, hs in per_layer.items()}

        logits = run_with_intervention(wrapper, native_texts[i], zero_interv)
        n0.append(margin_fn(logits))

        logits = run_with_intervention(wrapper, latin_texts[i], zero_interv)
        l0.append(margin_fn(logits))

        patch_interv_NP = {
            l: [(h, latin_caches[i]["act"][l][h]) for h in hs]
            for l, hs in per_layer.items()
        }
        logits = run_with_intervention(wrapper, native_texts[i], patch_interv_NP)
        np_.append(margin_fn(logits))

        patch_interv_LP = {
            l: [(h, native_caches[i]["act"][l][h]) for h in hs]
            for l, hs in per_layer.items()
        }
        logits = run_with_intervention(wrapper, latin_texts[i], patch_interv_LP)
        lp.append(margin_fn(logits))

    return {
        "N0": n0, "NP": np_, "L0": l0, "LP": lp,
        "baseline_N": bN, "baseline_L": bL,
    }


# ────────────────────────────────────────────────────────────────
# Orchestration
# ────────────────────────────────────────────────────────────────

def prepare_caches(wrapper, texts, label):
    out = {}
    for i, t in enumerate(tqdm(texts, desc=f"Caching {label}")):
        act, logits = cache_last_token_activations(wrapper, t)
        out[i] = {"act": act, "logits": logits}
    return out


def summarize_cell(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    return {
        "mean": float(a.mean()),
        "sem":  float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "n":    int(n),
    }


def summarize_group(result):
    s = {k: summarize_cell(v) for k, v in result.items()}
    s["effects"] = {
        "zero_native_drop":   s["baseline_N"]["mean"] - s["N0"]["mean"],
        "patch_native_drop":  s["baseline_N"]["mean"] - s["NP"]["mean"],
        "zero_latin_drop":    s["L0"]["mean"] - s["baseline_L"]["mean"],
        "patch_latin_drop":   s["LP"]["mean"] - s["baseline_L"]["mean"],
    }
    return s


# ────────────────────────────────────────────────────────────────
# Head selection
# ────────────────────────────────────────────────────────────────

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_diff_matrix(model_key, lang_code):
    path = os.path.join(results_dir(model_key, lang_code, "dla"), "diff.npy")
    return np.load(path)


def load_patch_symmetric(model_key, lang_code):
    path = os.path.join(results_dir(model_key, lang_code, "patch"), "effects_symmetric.npy")
    return np.load(path)


HEAD_SOURCES = {
    "dla_abs":     {"subdir": "dla",   "file": "top.json",                "key": "top_heads_by_abs_diff",      "matrix": "dla/diff.npy",               "use_abs": True},
    "dla_native":  {"subdir": "dla",   "file": "top_native_writers.json", "key": "top_heads_by_native_writing","matrix": "dla/diff.npy",               "use_abs": False, "sign": "neg"},
    "dla_latin":   {"subdir": "dla",   "file": "top_latin_writers.json",  "key": "top_heads_by_latin_writing", "matrix": "dla/diff.npy",               "use_abs": False, "sign": "pos"},
    "patch_abs":   {"subdir": "patch", "file": "top.json",                "key": "top_heads_by_abs_effect",    "matrix": "patch/effects_symmetric.npy","use_abs": True},
    "patch_n2l":   {"subdir": "patch", "file": "top_n2l.json",            "key": "top_heads_n2l_pos",          "matrix": "patch/effects_n2l.npy",      "use_abs": False, "sign": "pos"},
    "patch_l2n":   {"subdir": "patch", "file": "top_l2n.json",            "key": "top_heads_l2n_pos",          "matrix": "patch/effects_l2n.npy",      "use_abs": False, "sign": "pos"},
}


def load_top_heads(model_key, lang_code, k, source="dla_abs"):
    if source in ("intersection", "union"):
        a = load_top_heads(model_key, lang_code, k, "dla_abs")
        b = load_top_heads(model_key, lang_code, k, "patch_abs")
        sa, sb = set(a), set(b)
        if source == "intersection":
            return [h for h in a if h in sb]
        seen = set(a); out = list(a)
        for h in b:
            if h not in seen:
                out.append(h); seen.add(h)
        return out[:k]

    if source not in HEAD_SOURCES:
        raise ValueError(f"Unknown head source: {source}")
    spec = HEAD_SOURCES[source]
    path = os.path.join(results_dir(model_key, lang_code, spec["subdir"]), spec["file"])
    data = _load_json(path)
    items = data[spec["key"]][:k]
    return [(int(it["layer"]), int(it["head"])) for it in items]


def load_ranking_matrix(model_key, lang_code, source="dla_abs"):
    if source in ("intersection", "union"):
        return load_diff_matrix(model_key, lang_code)
    spec = HEAD_SOURCES[source]
    return np.load(os.path.join(RESULTS_DIR, model_key, lang_code, spec["matrix"]))


def same_layer_controls(top_heads, diff_matrix, n_per_head=1):
    chosen = set(top_heads)
    out = []
    for l, _ in top_heads:
        row = np.abs(diff_matrix[l])
        order = np.argsort(row)
        for h in order:
            h = int(h)
            if (l, h) not in chosen and len(out) < len(top_heads) * n_per_head:
                out.append((l, h))
                chosen.add((l, h))
                break
    return out


def global_bottom_heads(diff_matrix, k, exclude=()):
    flat = np.abs(diff_matrix).ravel()
    order = np.argsort(flat)
    out = []
    excl = set(exclude)
    for idx in order:
        l, h = np.unravel_index(int(idx), diff_matrix.shape)
        l, h = int(l), int(h)
        if (l, h) in excl:
            continue
        out.append((l, h))
        if len(out) >= k:
            break
    return out


def random_heads(diff_matrix, k, rng, exclude=()):
    excl = set(exclude)
    n_layers, n_heads = diff_matrix.shape
    out = []
    while len(out) < k:
        l = int(rng.integers(n_layers))
        h = int(rng.integers(n_heads))
        if (l, h) not in excl and (l, h) not in set(out):
            out.append((l, h))
    return out


# ────────────────────────────────────────────────────────────────
# Plotting (no titles)
# ────────────────────────────────────────────────────────────────

def plot_single_heads(summaries, order, out_path):
    set_icml_style()
    def _label(key):
        m = summaries[key]["_meta"]
        tag_short = {"top": "top", "same_layer": "SL",
                     "global_bottom": "bot", "random": "rand"}.get(m["tag"], m["tag"])
        return f"[{tag_short}] L{m['layer']}H{m['head']}"
    labels = [_label(k) for k in order]
    eff = lambda tag: [summaries[k]["effects"][tag] for k in order]

    x = np.arange(len(order))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(5, 0.65 * len(order) + 2), 3.2))
    pal = sns.color_palette("tab10")

    ax.bar(x - 1.5*w, eff("zero_native_drop"),  w, label="Zero (native in)",  color=pal[0], edgecolor="black", lw=0.6)
    ax.bar(x - 0.5*w, eff("patch_native_drop"), w, label="Patch←latin (native in)", color=pal[1], edgecolor="black", lw=0.6)
    ax.bar(x + 0.5*w, eff("zero_latin_drop"),   w, label="Zero (latin in)",   color=pal[2], edgecolor="black", lw=0.6)
    ax.bar(x + 1.5*w, eff("patch_latin_drop"),  w, label="Patch←native (latin in)", color=pal[3], edgecolor="black", lw=0.6)

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Δ margin (positive = script flipped)", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=7, ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_subsets(summaries, group_labels, out_path):
    set_icml_style()
    eff = lambda tag: [summaries[k]["effects"][tag] for k in group_labels]
    x = np.arange(len(group_labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(group_labels) + 2), 3.6))
    pal = sns.color_palette("tab10")

    ax.bar(x - 1.5*w, eff("zero_native_drop"),  w, label="Zero (native in)",  color=pal[0], edgecolor="black", lw=0.6)
    ax.bar(x - 0.5*w, eff("patch_native_drop"), w, label="Patch←latin (native in)", color=pal[1], edgecolor="black", lw=0.6)
    ax.bar(x + 0.5*w, eff("zero_latin_drop"),   w, label="Zero (latin in)",   color=pal[2], edgecolor="black", lw=0.6)
    ax.bar(x + 1.5*w, eff("patch_latin_drop"),  w, label="Patch←native (latin in)", color=pal[3], edgecolor="black", lw=0.6)

    def short(lbl):
        i = lbl.find("[")
        return lbl[:i] if i > 0 else lbl
    short_labels = [short(l) for l in group_labels]

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Δ margin", fontweight="bold")
    ax.legend(loc="best", framealpha=0.9, fontsize=7, ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ────────────────────────────────────────────────────────────────
# Attention visualization (no titles)
# ────────────────────────────────────────────────────────────────

def get_attention_weights(wrapper, text, layer_idx):
    inputs = wrapper.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        out = wrapper.model(**inputs, output_attentions=True)
    attn = out.attentions[layer_idx][0].to(torch.float32).cpu().numpy()
    tokens = wrapper.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    return attn, tokens


def plot_head_attention(wrapper, head_tuple, example_text, out_path, tag="native"):
    layer, head = head_tuple
    set_icml_style()
    attn, tokens = get_attention_weights(wrapper, example_text, layer)
    A = attn[head]
    T = len(tokens)
    labels = [t.replace("▁", "_").replace("Ġ", "_")[:8] for t in tokens]

    fig, ax = plt.subplots(figsize=(max(3.5, T*0.18), max(3.0, T*0.15)))
    sns.heatmap(A, ax=ax, cmap="Blues", vmin=0, vmax=A.max(),
                xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.7})
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_head_aggregate(wrapper, head_tuple, texts, out_path, tag=""):
    layer, head = head_tuple
    bos_atts, self_atts, dist_profiles = [], [], []

    for text in tqdm(texts, desc=f"Aggregating L{layer}H{head}", leave=False):
        attn, _ = get_attention_weights(wrapper, text, layer)
        A = attn[head]
        T = A.shape[0]
        if T > 1:
            bos_atts.append(A[1:, 0].mean())
            self_atts.append(np.diag(A).mean())

        prof = np.zeros(T); cnt = np.zeros(T)
        for q in range(T):
            for d in range(q + 1):
                prof[d] += A[q, q - d]; cnt[d] += 1
        dist_profiles.append(prof / np.maximum(cnt, 1))

    maxT = max(len(p) for p in dist_profiles)
    padded = np.full((len(dist_profiles), maxT), np.nan)
    for i, p in enumerate(dist_profiles):
        padded[i, :len(p)] = p
    mean_prof = np.nanmean(padded, axis=0)
    sem_prof  = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0))

    set_icml_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
    vals = [np.mean(bos_atts), np.mean(self_atts)]
    errs = [np.std(bos_atts)/np.sqrt(len(bos_atts)), np.std(self_atts)/np.sqrt(len(self_atts))]
    ax1.bar(["BOS", "Self"], vals, yerr=errs, color=sns.color_palette("tab10")[:2],
            edgecolor="black", linewidth=0.8, capsize=3)
    ax1.set_ylabel("Mean attention")

    k = np.arange(min(30, maxT))
    ax2.plot(k, mean_prof[:len(k)], lw=2, color=sns.color_palette("tab10")[1])
    ax2.fill_between(k, mean_prof[:len(k)] - sem_prof[:len(k)],
                     mean_prof[:len(k)] + sem_prof[:len(k)], alpha=0.25)
    ax2.set_xlabel("Distance (query − key)")
    ax2.set_ylabel("Mean attention")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ────────────────────────────────────────────────────────────────
# Experiment runners
# ────────────────────────────────────────────────────────────────

def run_single(wrapper, args, lang_cfg, model_cfg, top_heads, score_matrix,
               native_caches, latin_caches, native_texts, latin_texts, margin_fn,
               baseline_gens=None, source_tag="dla_abs"):
    rng = np.random.default_rng(SEED)
    same_layer = same_layer_controls(top_heads, score_matrix, n_per_head=1)
    excl = set(top_heads) | set(same_layer)
    bottom = global_bottom_heads(score_matrix, args.n_bottom_controls, exclude=excl)
    excl |= set(bottom)
    randoms = random_heads(score_matrix, args.n_random_controls, rng, exclude=excl)

    all_heads = [("top", top_heads),
                 ("same_layer", same_layer),
                 ("global_bottom", bottom),
                 ("random", randoms)]

    gen_tags = {"top", "random"} if args.save_generations else set()

    summaries, raw, generations = {}, {}, {}
    order = []
    for tag, heads in all_heads:
        for head in tqdm(heads, desc=f"Single-head sweep [{source_tag}/{tag}]"):
            key = f"{tag}/L{head[0]}H{head[1]}"
            res = measure_head_group(wrapper, [head], native_texts, latin_texts,
                                     native_caches, latin_caches, margin_fn)
            summaries[key] = summarize_group(res)
            summaries[key]["_meta"] = {"tag": tag, "layer": head[0], "head": head[1]}
            raw[key] = res
            order.append(key)

            if tag in gen_tags:
                gens = generate_for_group(
                    wrapper, [head], native_texts, latin_texts,
                    native_caches, latin_caches, lang_cfg["native_unicode_range"],
                    max_new_tokens=args.gen_max_new_tokens,
                    baseline_gens=baseline_gens,
                )
                generations[key] = gens
                summaries[key]["generation_summary"] = gens["summary"]

    out_dir = results_dir(args.model, args.lang, f"validation/{source_tag}/single")
    save_json({"summaries": summaries, "raw": raw, "order": order, "source": source_tag,
               "head_groups": {t: list(map(list, hs)) for t, hs in all_heads}},
              os.path.join(out_dir, "results.json"))
    if generations:
        save_json({"generations": generations},
                  os.path.join(out_dir, "generations.json"))
        print(f"Saved generations: {os.path.join(out_dir, 'generations.json')}")

    plot_single_heads(summaries, order,
                      os.path.join(plots_dir(args.model, args.lang, f"validation/{source_tag}"),
                                   "single_effects.pdf"))
    return summaries


def run_subsets(wrapper, args, lang_cfg, model_cfg, top_heads, score_matrix,
                native_caches, latin_caches, native_texts, latin_texts, margin_fn,
                baseline_gens=None, source_tag="dla_abs"):
    def fmt_heads(heads):
        return ",".join(f"L{l}H{h}" for l, h in heads)

    groups = {}

    for k in (2, 3, 5, 10):
        if k <= len(top_heads):
            groups[f"top-{k}[{fmt_heads(top_heads[:k])}]"] = top_heads[:k]

    if len(top_heads) >= 5:
        for a, b in combinations(range(5), 2):
            ha, hb = top_heads[a], top_heads[b]
            groups[f"pair[L{ha[0]}H{ha[1]},L{hb[0]}H{hb[1]}]"] = [ha, hb]

    if len(top_heads) >= 5:
        triple = top_heads[:3]
        groups[f"triple[{fmt_heads(triple)}]"] = triple

    rng = np.random.default_rng(SEED + 1)
    rand5 = random_heads(score_matrix, 5, rng, exclude=set(top_heads))
    groups[f"random-5[{fmt_heads(rand5)}]"] = rand5
    bot5 = global_bottom_heads(score_matrix, 5, exclude=set(top_heads))
    groups[f"bottom-5[{fmt_heads(bot5)}]"] = bot5

    summaries, raw, generations = {}, {}, {}
    order = list(groups.keys())
    for label in tqdm(order, desc=f"Subset sweep [{source_tag}]"):
        res = measure_head_group(wrapper, groups[label], native_texts, latin_texts,
                                 native_caches, latin_caches, margin_fn)
        summaries[label] = summarize_group(res)
        summaries[label]["_meta"] = {"heads": list(map(list, groups[label]))}
        raw[label] = res

        if args.save_generations:
            gens = generate_for_group(
                wrapper, groups[label], native_texts, latin_texts,
                native_caches, latin_caches, lang_cfg["native_unicode_range"],
                max_new_tokens=args.gen_max_new_tokens,
                baseline_gens=baseline_gens,
            )
            generations[label] = gens
            summaries[label]["generation_summary"] = gens["summary"]

    out_dir = results_dir(args.model, args.lang, f"validation/{source_tag}/subsets")
    save_json({"summaries": summaries, "raw": raw, "order": order, "source": source_tag,
               "groups": {k: list(map(list, v)) for k, v in groups.items()}},
              os.path.join(out_dir, "results.json"))
    if generations:
        save_json({"generations": generations},
                  os.path.join(out_dir, "generations.json"))
        print(f"Saved generations: {os.path.join(out_dir, 'generations.json')}")

    plot_subsets(summaries, order,
                 os.path.join(plots_dir(args.model, args.lang, f"validation/{source_tag}"),
                              "subsets_effects.pdf"))
    return summaries


def run_attention(wrapper, args, lang_cfg, top_heads):
    prompts = load_validation_prompts(args.lang, args.n_prompts)
    example_nat = prompts["native"][0]
    example_lat = prompts["latin"][0]
    attn_dir = plots_dir(args.model, args.lang, "attention")
    for head in top_heads[:args.k_viz]:
        l, h = head
        plot_head_attention(wrapper, head, example_nat, os.path.join(attn_dir, f"L{l}H{h}_native.pdf"), "native")
        plot_head_attention(wrapper, head, example_lat, os.path.join(attn_dir, f"L{l}H{h}_latin.pdf"),  "latin")
        plot_head_aggregate(wrapper, head, prompts["native"], os.path.join(attn_dir, f"L{l}H{h}_agg_native.pdf"), "native")
        plot_head_aggregate(wrapper, head, prompts["latin"],  os.path.join(attn_dir, f"L{l}H{h}_agg_latin.pdf"),  "latin")


# ────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ────────────────────────────────────────────────────────────────

def print_single_table(summaries, order):
    print("\n  Head                  |  zero_N  patch_N |  zero_L  patch_L | (all: positive = flipped script)")
    print("  " + "-" * 88)
    for key in order:
        s = summaries[key]["effects"]
        meta = summaries[key]["_meta"]
        name = f"{meta['tag']:>13s} L{meta['layer']:02d}H{meta['head']:02d}"
        print(f"  {name:<22s}|  {s['zero_native_drop']:+6.3f}  {s['patch_native_drop']:+6.3f} |"
              f"  {s['zero_latin_drop']:+6.3f}  {s['patch_latin_drop']:+6.3f}")


def print_subset_table(summaries, order):
    print("\n  Group                          |  zero_N  patch_N |  zero_L  patch_L")
    print("  " + "-" * 80)
    for key in order:
        s = summaries[key]["effects"]
        print(f"  {key:<30s} |  {s['zero_native_drop']:+6.3f}  {s['patch_native_drop']:+6.3f} |"
              f"  {s['zero_latin_drop']:+6.3f}  {s['patch_latin_drop']:+6.3f}")


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

def run(args):
    lang_cfg  = LANG_CONFIG[args.lang]
    model_cfg = MODEL_CONFIG[args.model]
    wrapper   = ModelWrapper(args.model)

    prompts = load_validation_prompts(args.lang, args.n_prompts)
    native_texts = prompts["native"]
    latin_texts  = prompts["latin"]
    assert len(native_texts) == len(latin_texts), "Prompt pairs must be parallel"
    print(f"Using {len(native_texts)} parallel prompt pairs from data/")

    vocab_size = wrapper.model.get_output_embeddings().weight.shape[0]
    print(f"Building token script masks (vocab={vocab_size})...")
    nat_mask, lat_mask = build_token_script_masks(
        wrapper.tokenizer, lang_cfg["native_unicode_range"], vocab_size
    )
    print(f"  native-script tokens: {nat_mask.sum()}, latin tokens: {lat_mask.sum()}")
    margin_fn = script_margin_fn(nat_mask, lat_mask)

    native_caches = prepare_caches(wrapper, native_texts, "native")
    latin_caches  = prepare_caches(wrapper, latin_texts,  "latin")

    baseline_gens = None
    if args.save_generations and args.experiment in ("single", "subsets", "all"):
        print("\nPrecomputing baseline generations (shared across all groups)...")
        baseline_gens = precompute_baseline_generations(
            wrapper, native_texts, latin_texts, args.gen_max_new_tokens,
        )

    if args.experiment in ("attention", "all"):
        print("\n====== ATTENTION VISUALIZATION ======")
        viz_heads = load_top_heads(args.model, args.lang, args.k, "dla_abs")
        run_attention(wrapper, args, lang_cfg, viz_heads)

    for source in args.head_source:
        print(f"\n{'='*64}")
        print(f"HEAD SOURCE: {source}")
        print('=' * 64)

        try:
            top_heads = load_top_heads(args.model, args.lang, args.k, source)
            score_matrix = load_ranking_matrix(args.model, args.lang, source)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping '{source}': {e}")
            continue

        print(f"Top-{args.k} heads [{source}]: {top_heads}")

        if args.experiment in ("single", "all"):
            print(f"\n------ SINGLE-HEAD SWEEP [{source}] ------")
            s = run_single(wrapper, args, lang_cfg, model_cfg, top_heads, score_matrix,
                           native_caches, latin_caches, native_texts, latin_texts, margin_fn,
                           baseline_gens=baseline_gens, source_tag=source)
            order = list(s.keys())
            print_single_table(s, order)

        if args.experiment in ("subsets", "all"):
            print(f"\n------ SUBSET SWEEP [{source}] ------")
            s = run_subsets(wrapper, args, lang_cfg, model_cfg, top_heads, score_matrix,
                            native_caches, latin_caches, native_texts, latin_texts, margin_fn,
                            baseline_gens=baseline_gens, source_tag=source)
            print_subset_table(s, list(s.keys()))


def main():
    p = argparse.ArgumentParser(description="Causal validation of script-mediation heads")
    p.add_argument("--model",      choices=MODEL_CONFIG, default="llama3-8b")
    p.add_argument("--lang",       choices=LANG_CONFIG,  default="hi")
    p.add_argument("--experiment", choices=["single", "subsets", "attention", "all"], default="all")
    p.add_argument("--k",          type=int, default=10, help="Top-k heads to validate")
    p.add_argument("--k-viz",      type=int, default=3,  help="Heads to visualize attention for")
    p.add_argument("--n-prompts",  type=int, default=10,
                   help="Parallel prompt pairs to load from data/<lang>.txt and data/<lang>_rom.txt")
    p.add_argument("--n-bottom-controls", type=int, default=3)
    p.add_argument("--n-random-controls", type=int, default=3)
    p.add_argument("--save-generations", action="store_true",
                   help="Also generate completions under each intervention condition.")
    p.add_argument("--gen-max-new-tokens", type=int, default=60)
    p.add_argument(
        "--head-source", nargs="+",
        default=["patch_abs"],
        choices=list(HEAD_SOURCES.keys()) + ["intersection", "union"],
        help="Which head ranking(s) to validate. Multiple allowed.",
    )
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run(args)


if __name__ == "__main__":
    main()