import os
import json
import glob
import argparse
import copy
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

DATA_DIR = "data"
ACTIVATION_DIR = "activations"
ACTIVATION_SPLIT = "dev"
OUTPUT_DIR = "results"

MODEL_CONFIG = {
    "llama3-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "layers_attr": ["model", "layers"],
        "n_layers": 32,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        "extra_generate_kwargs": {},
    },
    "llama3-70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "layers_attr": ["model", "layers"],
        "n_layers": 80,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        "extra_generate_kwargs": {},
    },
    "aya-8b": {
        "model_id": "CohereForAI/aya-expanse-8b",
        "layers_attr": ["model", "layers"],
        "n_layers": 32,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        "extra_generate_kwargs": {},
    },
    "aya-32b": {
        "model_id": "CohereForAI/aya-expanse-32b",
        "layers_attr": ["model", "layers"],
        "n_layers": 64,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
        "extra_generate_kwargs": {},
    },
}

# Activation files for languages that have steering vectors derived
STEERING_VECTORS = {
    "hi": ("flores_hin_Deva.pt", "flores_hin_Latn.pt"),
    "ar": ("flores_arb_Arab.pt", "flores_arb_Latn.pt"),
}

# Files in data/ that are not language prompt files
DATA_NON_LANG_FILES = {"romanization.txt", "get.py"}


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def get_model_layers(model, layers_attr):
    obj = model
    for attr in layers_attr:
        obj = getattr(obj, attr)
    return obj


def discover_language_files(data_dir=DATA_DIR):
    """Find all data/<lang>.txt files, excluding romanization and helpers."""
    files = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        name = os.path.basename(path)
        if name in DATA_NON_LANG_FILES:
            continue
        if name.endswith("_rom.txt"):
            continue
        lang = name[:-len(".txt")]
        files[lang] = path
    return files


def load_prompts(path, max_prompts=None):
    with open(path, encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    return prompts[:max_prompts] if max_prompts else prompts


# ────────────────────────────────────────────────────────────────
# Vector building
# ────────────────────────────────────────────────────────────────

def load_single_diff(model_key, lang_code):
    """Returns dict layer -> raw (latin - native) diff (cpu, float32)."""
    if lang_code not in STEERING_VECTORS:
        raise ValueError(f"No steering vector files configured for {lang_code}")
    native_file, latin_file = STEERING_VECTORS[lang_code]
    acts_dir = os.path.join(ACTIVATION_DIR, model_key, ACTIVATION_SPLIT)
    acts_native = torch.load(os.path.join(acts_dir, native_file))
    acts_latin = torch.load(os.path.join(acts_dir, latin_file))
    return {k: (acts_latin[k] - acts_native[k]).float() for k in acts_native}


def combine_vectors(per_lang_diffs, combination):
    """per_lang_diffs: list of dicts (layer -> tensor). Returns dict (layer -> tensor)."""
    layers = list(per_lang_diffs[0].keys())
    combined = {}
    for layer in layers:
        stacked = torch.stack([d[layer] for d in per_lang_diffs])
        if combination == "mean":
            v = stacked.mean(dim=0)
        elif combination == "sum":
            v = stacked.sum(dim=0)
        elif combination == "pca":
            centered = stacked - stacked.mean(dim=0, keepdim=True)
            _, _, V = torch.svd(centered)
            v = V[:, 0]
        else:
            raise ValueError(f"Unknown combination method: {combination}")
        combined[layer] = v
    return combined


def build_steering_vectors(model_key, vector_sources, combination, device):
    """Returns {'nat2lat': {layer: vec}, 'lat2nat': {layer: vec}} (unit-norm)."""
    diffs = [load_single_diff(model_key, src) for src in vector_sources]
    combined = combine_vectors(diffs, combination)

    nat2lat, lat2nat = {}, {}
    for layer, v in combined.items():
        v = v / (torch.norm(v) + 1e-8)
        v = v.to(device).to(torch.bfloat16)
        nat2lat[layer] = v
        lat2nat[layer] = -v
    return {"nat2lat": nat2lat, "lat2nat": lat2nat}


# ────────────────────────────────────────────────────────────────
# Tester
# ────────────────────────────────────────────────────────────────

class GeneralityTester:
    def __init__(self, model_key, vector_sources, combination):
        cfg = MODEL_CONFIG[model_key]
        self.model_key = model_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generate_kwargs = copy.deepcopy(cfg.get("extra_generate_kwargs", {}))
        self.n_layers = cfg["n_layers"]
        self.vector_sources = vector_sources
        self.combination = combination

        print(f"Loading model: {cfg['model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], padding_side="left", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.layers = get_model_layers(self.model, cfg["layers_attr"])

        self.vectors_by_dir = build_steering_vectors(
            model_key, vector_sources, combination, self.device
        )
        # Move each layer's vector to the device of that layer (handles multi-GPU)
        self._move_vectors_to_layer_devices()
        print(f"Built vectors from sources={vector_sources}, combine={combination}")

    def _layer_device(self, idx):
        """Get the device of layer `idx` (works with device_map='auto' sharding)."""
        try:
            return next(self.layers[idx].parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def _move_vectors_to_layer_devices(self):
        for direction, vecs in self.vectors_by_dir.items():
            for i in range(self.n_layers):
                key = f"layer_{i}"
                if key in vecs:
                    vecs[key] = vecs[key].to(self._layer_device(i))

    def _make_hook(self, vec, scale):
        def hook(_, __, output):
            h = output[0] if isinstance(output, tuple) else output
            v = vec
            if v.device != h.device or v.dtype != h.dtype:
                v = v.to(device=h.device, dtype=h.dtype)
            h = h + scale * v
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook

    def generate_batch(self, prompts, direction, layer_indices, scale, max_tokens=100):
        """Batched steered generation. Returns list of decoded continuations."""
        handles = []
        if scale != 0.0:
            vectors = self.vectors_by_dir[direction]
            indices = layer_indices if layer_indices is not None else list(range(self.n_layers))
            for idx in indices:
                vec = vectors[f"layer_{idx}"]
                handles.append(
                    self.layers[idx].register_forward_hook(self._make_hook(vec, scale))
                )

        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generate_kwargs,
                )
            prompt_len = inputs["input_ids"].shape[1]
            return [
                self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
                for o in outputs
            ]
        finally:
            for h in handles:
                h.remove()


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

def output_path(model_key, vector_sources, combination, layer, scale, direction):
    out_dir = os.path.join(OUTPUT_DIR, model_key, "generality")
    os.makedirs(out_dir, exist_ok=True)
    sources_str = "+".join(vector_sources)
    layer_tag = "all" if layer is None else f"L{layer}"
    fname = f"{sources_str}_{combination}_{layer_tag}_S{scale}_{direction}.json"
    return os.path.join(out_dir, fname)


def run_one_setting(tester, languages, prompts_by_lang, direction, layer_indices, scale, max_tokens):
    """Run baseline + steered for every prompt in every language."""
    print(f"\n### {direction} | layers={layer_indices if layer_indices else 'all'} | scale={scale} ###")

    out = {
        "config": {
            "model": tester.model_key,
            "vector_sources": tester.vector_sources,
            "combination": tester.combination,
            "direction": direction,
            "layer_indices": layer_indices,
            "scale": scale,
            "max_tokens": max_tokens,
            "timestamp": datetime.now().isoformat(),
        },
        "results": {},
    }

    for lang in tqdm(languages, desc=f"{direction} S={scale}"):
        prompts = prompts_by_lang[lang]
        baselines = tester.generate_batch(prompts, direction, layer_indices, 0.0, max_tokens)
        steered = tester.generate_batch(prompts, direction, layer_indices, scale, max_tokens)
        out["results"][lang] = [
            {"prompt": p, "baseline": b, "steered": s}
            for p, b, s in zip(prompts, baselines, steered)
        ]

    return out


def run(args):
    cfg = MODEL_CONFIG[args.model]
    vector_sources = args.vectors.split(",")

    # Auto-discover all languages or filter by --langs
    available = discover_language_files()
    if args.langs:
        wanted = args.langs.split(",")
        languages = []
        for lang in wanted:
            if lang in available:
                languages.append(lang)
            else:
                print(f"Warning: data/{lang}.txt not found, skipping")
    else:
        languages = list(available.keys())

    prompts_by_lang = {
        lang: load_prompts(available[lang], args.max_prompts)
        for lang in languages
    }
    languages = [l for l in languages if prompts_by_lang[l]]
    print(f"Testing {len(languages)} languages: {languages}")

    tester = GeneralityTester(args.model, vector_sources, args.combine)

    # Decide layer config and scales
    if args.all_layers:
        layer_indices = None      # signals "all layers" inside generate()
        layer_tag = None
        scales = cfg["all_layers_scaling_factors"]
    else:
        if args.layer is None:
            raise ValueError("Must specify --layer when not using --all-layers")
        layer_indices = [args.layer]
        layer_tag = args.layer
        scales = cfg["scaling_factors"]

    # If user passed explicit --scale, override the sweep
    if args.scale is not None:
        scales = [args.scale]

    directions = ["nat2lat", "lat2nat"] if args.both_directions else [args.direction]

    for direction in directions:
        for scale in scales:
            results = run_one_setting(
                tester, languages, prompts_by_lang,
                direction, layer_indices, scale, args.max_tokens,
            )
            path = output_path(args.model, vector_sources, args.combine,
                               layer_tag, scale, direction)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Test steering vector generality across languages")
    parser.add_argument("--model", choices=MODEL_CONFIG, default="llama3-8b")
    parser.add_argument("--vectors", default="hi,ar",
                        help="Comma-separated source languages for steering vector")
    parser.add_argument("--combine", choices=["mean", "sum", "pca"], default="mean")
    parser.add_argument("--all-layers", action="store_true",
                        help="Apply steering to all layers (uses finer-grained scales)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer (required if not --all-layers)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Single scale; if omitted, sweeps configured scales")
    parser.add_argument("--direction", choices=["nat2lat", "lat2nat"], default="nat2lat",
                        help="Steering direction (ignored if --both-directions)")
    parser.add_argument("--both-directions", action="store_true", default=True,
                        help="Run both nat2lat and lat2nat")
    parser.add_argument("--langs", type=str, default=None,
                        help="Comma-separated languages to test (default: all in data/)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Cap prompts per language file")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()