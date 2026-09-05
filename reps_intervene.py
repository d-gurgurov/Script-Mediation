import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import re
from tqdm import tqdm
import argparse
import copy

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

DATA_DIR = "data"
OUTPUT_DIR = "results"
ACTIVATION_DIR = "activations"
ACTIVATION_SPLIT = "dev"

BATCH_SIZE = 70
DIRECTIONS = ["nat2lat", "lat2nat"]

MODEL_CONFIG = {
    "llama3-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "layers_attr": ["model", "layers"],
        "n_layers": 32,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0],
        "extra_generate_kwargs": {},
    },
    "llama3-70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "layers_attr": ["model", "layers"],
        "n_layers": 80,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0],
        "extra_generate_kwargs": {},
    },
    "aya-8b": {
        "model_id": "CohereForAI/aya-expanse-8b",
        "layers_attr": ["model", "layers"],
        "n_layers": 32,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0],
        "extra_generate_kwargs": {},
    },
    "aya-32b": {
        "model_id": "CohereForAI/aya-expanse-32b",
        "layers_attr": ["model", "layers"],
        "n_layers": 40,
        "scaling_factors": [0.0, 1.0, 2.5, 5.0, 10.0],
        "all_layers_scaling_factors": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0],
        "extra_generate_kwargs": {},
    },
}

LANG_CONFIG = {
    "hi": {
        "name": "Hindi",
        "native_acts": "flores_hin_Deva.pt",
        "latin_acts": "flores_hin_Latn.pt",
        "native_regex": r"[\u0900-\u097F]",
        "native_prompts": "hi.txt",
        "latin_prompts": "hi_rom.txt",
    },
    "ar": {
        "name": "Arabic",
        "native_acts": "flores_arb_Arab.pt",
        "latin_acts": "flores_arb_Latn.pt",
        "native_regex": r"[\u0600-\u06FF]",
        "native_prompts": "ar.txt",
        "latin_prompts": "ar_rom.txt",
    },
}


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def get_model_layers(model, layers_attr):
    obj = model
    for attr in layers_attr:
        obj = getattr(obj, attr)
    return obj


def load_prompts(path, max_prompts=None):
    with open(path, encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    return prompts[:max_prompts] if max_prompts else prompts


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def format_chat_prompts(tokenizer, prompts, *, enable_thinking=None):
    texts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        texts.append(tokenizer.apply_chat_template(messages, **kwargs))
    return texts


def direction_prompts_file(direction, lang_cfg):
    """Pick prompt file based on the source script of the steering direction."""
    return lang_cfg["native_prompts"] if direction == "nat2lat" else lang_cfg["latin_prompts"]


# ────────────────────────────────────────────────────────────────
# Steering vector loading (both directions)
# ────────────────────────────────────────────────────────────────

def load_steering_vectors(model_key, lang_cfg, device):
    """Load activations once, build both nat2lat and lat2nat vectors.

    nat2lat = (latin - native), normalized
    lat2nat = -(nat2lat) = (native - latin), normalized
    """
    acts_dir = os.path.join(ACTIVATION_DIR, model_key, ACTIVATION_SPLIT)
    acts_native = torch.load(os.path.join(acts_dir, lang_cfg["native_acts"]))
    acts_latin = torch.load(os.path.join(acts_dir, lang_cfg["latin_acts"]))

    nat2lat = {}
    lat2nat = {}
    for k in acts_native:
        diff = (acts_latin[k] - acts_native[k]).to(device).to(torch.bfloat16)
        v = diff / (torch.norm(diff) + 1e-8)
        nat2lat[k] = v
        lat2nat[k] = -v

    return {"nat2lat": nat2lat, "lat2nat": lat2nat}


def save_steering_vectors(vectors_by_dir, model_key, lang_code):
    """Persist both direction vectors for downstream reuse."""
    out_dir = os.path.join(ACTIVATION_DIR, model_key, ACTIVATION_SPLIT, "steering_vectors")
    os.makedirs(out_dir, exist_ok=True)
    for direction, vecs in vectors_by_dir.items():
        path = os.path.join(out_dir, f"{lang_code}_{direction}.pt")
        # Move to CPU float32 for stable serialization
        cpu_vecs = {k: v.detach().cpu().float() for k, v in vecs.items()}
        torch.save(cpu_vecs, path)
        print(f"Saved steering vectors → {path}")


# ────────────────────────────────────────────────────────────────
# Evaluator
# ────────────────────────────────────────────────────────────────

class BatchedSteeringEvaluator:
    def __init__(self, model_key, lang_cfg):
        cfg = MODEL_CONFIG[model_key]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lang_cfg = lang_cfg

        extra_kwargs = copy.deepcopy(cfg.get("extra_generate_kwargs", {}))
        self.enable_thinking = extra_kwargs.pop("enable_thinking", None)
        self.generate_kwargs = extra_kwargs

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
        self.layers = get_model_layers(self.model, cfg["layers_attr"])

        self.vectors_by_dir = load_steering_vectors(model_key, lang_cfg, self.device)
        self.active_vectors = None  # set per direction

    def set_direction(self, direction):
        if direction not in self.vectors_by_dir:
            raise ValueError(f"Unknown direction: {direction}")
        self.active_vectors = self.vectors_by_dir[direction]

    def _make_hook(self, vec, scale):
        def hook(_, __, output):
            h = output[0] if isinstance(output, tuple) else output
            h += scale * vec
            return (h,) + output[1:] if isinstance(output, tuple) else h
        return hook

    def generate_batch_steered(self, layer_indices, scale, prompts):
        handles = []
        if scale != 0.0:
            for idx in layer_indices:
                vec = self.active_vectors[f"layer_{idx}"]
                handle = self.layers[idx].register_forward_hook(
                    self._make_hook(vec, scale)
                )
                handles.append(handle)

        chat_texts = format_chat_prompts(
            self.tokenizer, prompts, enable_thinking=self.enable_thinking
        )
        inputs = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generate_kwargs,
            )

        for h in handles:
            h.remove()

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True).strip()
            for o in outputs
        ]

    def generate_all_steered(self, layer_indices, scale, prompts, batch_size):
        out = []
        for batch in chunked(prompts, batch_size):
            out.extend(self.generate_batch_steered(layer_indices, scale, batch))
        return out


# ────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────

def get_layers_to_test(model_key):
    return list(range(0, MODEL_CONFIG[model_key]["n_layers"], 2))


def results_path(model_key, lang_code, direction, mode_suffix):
    out_dir = os.path.join(OUTPUT_DIR, model_key, lang_code)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"steering_{direction}_{mode_suffix}.json")


def run_direction(evaluator, model_key, lang_code, direction, prompts, all_layers, batch_size):
    evaluator.set_direction(direction)
    cfg = MODEL_CONFIG[model_key]
    n_layers = cfg["n_layers"]

    if all_layers:
        layer_configs = [list(range(n_layers))]
        scales = cfg["all_layers_scaling_factors"]
        mode_suffix = "all_layers"
    else:
        layer_configs = [[l] for l in get_layers_to_test(model_key)]
        scales = cfg["scaling_factors"]
        mode_suffix = "per_layer"

    print(f"\n=== {direction} | {mode_suffix} | {len(prompts)} prompts ===")

    results = {f"prompt_{i}": [] for i in range(len(prompts))}
    pbar = tqdm(total=len(layer_configs) * len(scales), desc=direction)
    for layer_set in layer_configs:
        for scale in scales:
            outs = evaluator.generate_all_steered(layer_set, scale, prompts, batch_size)
            for i, txt in enumerate(outs):
                results[f"prompt_{i}"].append({
                    "layers": layer_set,
                    "scale": scale,
                    "direction": direction,
                    "output_text": txt,
                })
            pbar.update(1)
    pbar.close()

    out_path = results_path(model_key, lang_code, direction, mode_suffix)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")


def run_steering(model_key, lang_code, max_prompts, batch_size, all_layers):
    lang_cfg = LANG_CONFIG[lang_code]

    evaluator = BatchedSteeringEvaluator(model_key, lang_cfg)
    save_steering_vectors(evaluator.vectors_by_dir, model_key, lang_code)

    for direction in DIRECTIONS:
        prompts_file = direction_prompts_file(direction, lang_cfg)
        prompts_path = os.path.join(DATA_DIR, prompts_file)
        if not os.path.exists(prompts_path):
            print(f"Warning: {prompts_path} not found, skipping {direction}")
            continue
        prompts = load_prompts(prompts_path, max_prompts)
        run_direction(evaluator, model_key, lang_code, direction,
                      prompts, all_layers, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIG, default="llama3-8b")
    parser.add_argument("--lang", choices=LANG_CONFIG, default="hi")
    parser.add_argument("--max-prompts", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--all-layers", action="store_true",
                        help="Apply steering to all layers simultaneously with finer-grained scales")
    args = parser.parse_args()

    run_steering(args.model, args.lang, args.max_prompts, args.batch_size, args.all_layers)


if __name__ == "__main__":
    main()