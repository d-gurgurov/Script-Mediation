import argparse
import glob
import json
import os
import re

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"

LANG_CONFIG = {
    "hi": {"name": "Hindi", "native_prompts": "hi.txt", "latin_prompts": "hi_rom.txt"},
    "ar": {"name": "Arabic", "native_prompts": "ar.txt", "latin_prompts": "ar_rom.txt"},
}

FILENAME_RE = re.compile(r"steering_(nat2lat|lat2nat)_all_layers_lid\.json$")

# ────────────────────────────────────────────────────────────────
# Judge prompt
# ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an impartial judge. Your task is to evaluate a multilingual response from a language steering experiment. A question was asked in {source_language}, and the model was steered to respond in {target_language}.

Important: do NOT evaluate *which language* the response is in or whether steering succeeded. Ignore the target language entirely for scoring - evaluate the final response solely on its content, regardless of the language used.

You must evaluate the response based on two criteria:
1.  **Relevance:** Is the response a relevant answer to the question?
2.  **Coherence:** Is the response coherent text, or is it gibberish, repetitive, or useless?

Focus only on these criteria and do not consider language conformity or steering success.

**Scoring:**
- **0:** The response is completely unrelated to the question OR it is total gibberish/useless text.
- **1:** The response is *somewhat* related to the question but may be incomplete, partially off-target, or minimally useful.
- **2:** The response is clearly and directly related to the question and is coherent and useful text.

Begin your evaluation with a brief explanation (a few sentences) of your reasoning.
After your explanation, provide the rating in this exact format: "Rating: [[score]]".
"""

USER_PROMPT_TEMPLATE = """Question ({source_language}):
{question}

Response ({target_language}):
{response}
"""


def parse_rating(text: str) -> int | None:
    for pattern in (r"Rating: \[\[([012])\]\]", r"\[\[([012])\]\]", r"\b([012])\b\s*$"):
        match = re.search(pattern, text.strip())
        if match:
            return int(match.group(1))
    print(f"Warning: could not parse rating from: {text[:100]}...")
    return None


def create_chat_messages(system_prompt, user_prompts):
    return [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": u}]
        for u in user_prompts
    ]


def load_judge(model_id, max_model_len, gpu_memory_utilization):
    print(f"Loading judge model: {model_id}")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()
    stop_strings = ["<|im_end|>", "<|endoftext|>"]
    stop_ids = [tokenizer.eos_token_id]
    for s in stop_strings:
        stop_ids.extend(tokenizer.encode(s, add_special_tokens=False))
    stop_ids = list(set(stop_ids))
    return llm, stop_ids


def judge_sampling_params(stop_ids, temperature, max_tokens):
    return SamplingParams(
        temperature=temperature,  # 0.0 = deterministic greedy
        top_p=1.0,
        max_tokens=max_tokens,
        seed=0,
        stop_token_ids=stop_ids,
    )


def run_judge(llm, sampling_params, source_lang, target_lang, trials, batch_size):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(source_language=source_lang, target_language=target_lang)
    user_prompts = [
        USER_PROMPT_TEMPLATE.format(
            source_language=source_lang,
            question=t["input"],
            target_language=target_lang,
            response=t["output"],
        )
        for t in trials
    ]
    judge_texts = []
    for i in range(0, len(user_prompts), batch_size):
        batch = user_prompts[i:i + batch_size]
        messages_batch = create_chat_messages(system_prompt, batch)
        outputs = llm.chat(messages_batch, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False})
        judge_texts.extend(o.outputs[0].text.strip() for o in outputs)
    return judge_texts


# ────────────────────────────────────────────────────────────────
# LaBSE cosine similarity
# ────────────────────────────────────────────────────────────────

def load_embedder(model_id="sentence-transformers/LaBSE"):
    print(f"Loading embedding model: {model_id}")
    return SentenceTransformer(model_id)


def cosine_similarities(embedder, questions, responses, batch_size=64):
    q_emb = embedder.encode(questions, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
    r_emb = embedder.encode(responses, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
    return (q_emb * r_emb).sum(dim=-1).cpu().tolist()


# ────────────────────────────────────────────────────────────────
# Reconstruct prompts + flatten trials from reps_intervene.py output
# ────────────────────────────────────────────────────────────────

def load_prompts(lang_code, direction):
    cfg = LANG_CONFIG[lang_code]
    filename = cfg["native_prompts"] if direction == "nat2lat" else cfg["latin_prompts"]
    path = os.path.join(DATA_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def language_labels(lang_code, direction):
    name = LANG_CONFIG[lang_code]["name"]
    native_label, latin_label = f"{name} (native script)", f"{name} (Latinized)"
    return (native_label, latin_label) if direction == "nat2lat" else (latin_label, native_label)


def flatten_trials(data, prompts):
    """{prompt_i: [{layers, scale, direction, output_text, ...lid fields}, ...]} -> flat list
    of trial dicts with 'input'/'output' attached in place (mutates dicts inside `data`)."""
    flat = []
    for prompt_i, trials in data.items():
        idx = int(prompt_i.split("_")[1])
        prompt_text = prompts[idx] if idx < len(prompts) else ""
        for t in trials:
            t["input"] = prompt_text
            t["output"] = t.get("output_text", "")
            flat.append(t)
    return flat


def parse_path(file_path):
    match = FILENAME_RE.search(file_path)
    if not match:
        return None
    direction = match.group(1)
    model_key, lang_code = os.path.normpath(file_path).split(os.sep)[-3:-1]
    return model_key, lang_code, direction, "all_layers"


# ────────────────────────────────────────────────────────────────
# File-level processing
# ────────────────────────────────────────────────────────────────

def process_file(file_path, llm, sampling_params, embedder, judge_batch_size, embed_batch_size):
    parsed = parse_path(file_path)
    if parsed is None:
        print(f"Skipping (unrecognized filename): {file_path}")
        return None
    model_key, lang_code, direction, mode_suffix = parsed

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = load_prompts(lang_code, direction)
    trials = flatten_trials(data, prompts)
    if not trials:
        print(f"Skipping (no trials): {file_path}")
        return None

    source_lang, target_lang = language_labels(lang_code, direction)
    judge_texts = run_judge(llm, sampling_params, source_lang, target_lang, trials, judge_batch_size)
    sims = cosine_similarities(embedder, [t["input"] for t in trials], [t["output"] for t in trials], embed_batch_size)

    for t, judge_text, sim in zip(trials, judge_texts, sims):
        t["judge_evaluation"] = {"judge_response": judge_text, "judge_score": parse_rating(judge_text)}
        t["cosine_similarity"] = sim

    return data, trials


def summarize_by_alpha(trials):
    per_alpha = {}
    for t in trials:
        bucket = per_alpha.setdefault(t["scale"], {"judge_scores": [], "cosine_sims": []})
        if t["judge_evaluation"]["judge_score"] is not None:
            bucket["judge_scores"].append(t["judge_evaluation"]["judge_score"])
        bucket["cosine_sims"].append(t["cosine_similarity"])

    return {
        alpha: {
            "avg_judge_score": sum(v["judge_scores"]) / len(v["judge_scores"]) if v["judge_scores"] else None,
            "avg_cosine_similarity": sum(v["cosine_sims"]) / len(v["cosine_sims"]) if v["cosine_sims"] else None,
            "n": len(v["cosine_sims"]),
        }
        for alpha, v in sorted(per_alpha.items())
    }


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quality assessment: LLM-as-judge (vLLM) + LaBSE cosine similarity, per alpha")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/LaBSE")
    parser.add_argument("--input_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results_quality")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 for deterministic greedy judging")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--judge_batch_size", type=int, default=70)
    parser.add_argument("--embed_batch_size", type=int, default=70)
    args = parser.parse_args()

    llm, stop_ids = load_judge(args.judge_model, args.max_model_len, args.gpu_memory_utilization)
    sampling_params = judge_sampling_params(stop_ids, args.temperature, args.max_tokens)
    embedder = load_embedder(args.embed_model)

    json_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "steering_*_all_layers_lid.json"), recursive=True))
    print(f"Found {len(json_files)} files to evaluate.")

    for file_path in tqdm(json_files, desc="Evaluating files"):
        result = process_file(file_path, llm, sampling_params, embedder, args.judge_batch_size, args.embed_batch_size)
        if result is None:
            continue
        data, trials = result

        summary = summarize_by_alpha(trials)
        data["quality_summary"] = summary
        print(f"{file_path}:")
        for alpha, s in summary.items():
            print(f"  alpha={alpha}: judge={s['avg_judge_score']}, cos_sim={s['avg_cosine_similarity']} (n={s['n']})")

        relative_path = os.path.relpath(file_path, args.input_dir)
        out_path = os.path.join(args.output_dir, relative_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()