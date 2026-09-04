import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
SPLITS = ["dev", "devtest"]
SAVE_DIR = "activations"

LANG_CONFIG = {
    "hi": {
        "name": "Hindi",
        "files": ["flores_hin_Deva.txt", "flores_hin_Latn.txt"],
    },
    "ar": {
        "name": "Arabic",
        "files": ["flores_arb_Arab.txt", "flores_arb_Latn.txt"],
    },
}

MODEL_CONFIG = {
    "llama3-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "layers_attr": ["model", "layers"],
    },
    "llama3-70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "layers_attr": ["model", "layers"],
    },
    "aya-8b": {
        "model_id": "CohereForAI/aya-expanse-8b",
        "layers_attr": ["model", "layers"],
    },
    "aya-32b": {
        "model_id": "CohereForAI/aya-expanse-32b",
        "layers_attr": ["model", "layers"],
    },
}


def get_model_layers(model, layers_attr):
    obj = model
    for attr in layers_attr:
        obj = getattr(obj, attr)
    return obj


class ActivationExtractor:
    def __init__(self, model_key, pooling_strategy="mean"):
        if model_key not in MODEL_CONFIG:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIG.keys())}")

        config = MODEL_CONFIG[model_key]
        self.model_key = model_key
        self.pooling_strategy = pooling_strategy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading model: {config['model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        
        # Ensure tokenizer has a chat template if instruct pooling is selected
        if self.pooling_strategy == "last_token_instruct" and not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("The selected model tokenizer does not support chat templates for 'last_token_instruct'.")

        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.layers = get_model_layers(self.model, config["layers_attr"])
        self.n_layers = len(self.layers)
        self.activations = {}

        print(f"Model loaded: {self.n_layers} layers")

    def get_hook(self, name):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states = hidden_states.detach().cpu().float()  # Shape: [1, seq_len, hidden_dim]
            
            if self.pooling_strategy == "mean":
                # Average across the sequence dimension
                pooled_act = hidden_states.mean(dim=1).squeeze(0)
            elif self.pooling_strategy in ["last_token", "last_token_instruct"]:
                # Extract the hidden state of the very last token
                pooled_act = hidden_states[0, -1, :]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

            self.activations.setdefault(name, []).append(pooled_act)
        return hook

    def run_extraction(self, file_path):
        self.activations = {}

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        hooks = []
        for i, layer in enumerate(self.layers):
            hooks.append(layer.register_forward_hook(self.get_hook(f"layer_{i}")))

        print(f"Extracting activations ({self.pooling_strategy}) from: {file_path}")
        for text in tqdm(lines):
            # If instruct template strategy is picked, format it as a user message
            if self.pooling_strategy == "last_token_instruct":
                messages = [{"role": "user", "content": text}]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)

        for h in hooks:
            h.remove()

        final_vectors = {
            layer: torch.stack(acts).mean(dim=0)
            for layer, acts in self.activations.items()
        }
        return final_vectors


def extract_for_split(extractor, model_key, split, lang_code):
    if lang_code not in LANG_CONFIG:
        raise ValueError(f"Unknown language: {lang_code}. Available: {list(LANG_CONFIG.keys())}")

    config = LANG_CONFIG[lang_code]
    save_dir = os.path.join(SAVE_DIR, model_key, split)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n--- {config['name']} | split={split} ---")

    for filename in config["files"]:
        input_path = os.path.join(split, filename)

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue

        vectors = extractor.run_extraction(input_path)

        save_name = filename.replace(".txt", ".pt")
        save_path = os.path.join(save_dir, save_name)
        torch.save(vectors, save_path)
        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract activations for steering vectors")
    parser.add_argument("--model", type=str, default="llama3-8b",
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--lang", type=str, default=None,
                        choices=list(LANG_CONFIG.keys()),
                        help="Single language (omit for all)")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                        choices=SPLITS, help="Which splits to process")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "last_token", "last_token_instruct"],
                        help="Strategy to aggregate token activations for a sentence")
    args = parser.parse_args()

    print(f"\n=== Extracting with {args.model} | pooling={args.pooling} | splits={args.splits} ===\n")
    extractor = ActivationExtractor(args.model, pooling_strategy=args.pooling)

    lang_codes = [args.lang] if args.lang else list(LANG_CONFIG.keys())

    for split in args.splits:
        for lang_code in lang_codes:
            extract_for_split(extractor, args.model, split, lang_code)


if __name__ == "__main__":
    main()