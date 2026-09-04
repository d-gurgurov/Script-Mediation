import os
import json
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
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42
FLIP_THRESHOLD = 0.5

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

# native_ranges: unicode ranges for char-ratio metric
# native_script: GlotLID script suffix for the language's native script
# latin_script:  GlotLID script suffix for romanized form (always 'Latn')
LANG_CONFIG = {
    "ru": {"name": "Russian",  "native_ranges": [(0x0400, 0x04FF)],
           "native_script": "Cyrl"},
    "mr": {"name": "Marathi",  "native_ranges": [(0x0900, 0x097F)],
           "native_script": "Deva"},
    "ur": {"name": "Urdu",     "native_ranges": [(0x0600, 0x06FF), (0xFB50, 0xFDFF)],
           "native_script": "Arab"},
    "el": {"name": "Greek",    "native_ranges": [(0x0370, 0x03FF)],
           "native_script": "Grek"},
    "ja": {"name": "Japanese", "native_ranges": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FFF)],
           "native_script": "Jpan"},
    "th": {"name": "Thai",     "native_ranges": [(0x0E00, 0x0E7F)],
           "native_script": "Thai"},
    "zh": {"name": "Chinese",  "native_ranges": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
           "native_script": "Hans"},
    "en_cy": {"name": "English (Cyrillic transliteration)",
              "native_ranges": [(0x0400, 0x04FF)], "native_script": "Cyrl"},
    "es": {"name": "Spanish",  "native_ranges": [], "native_script": None},
}

# (PROMPTS dict copied verbatim from the original script)
PROMPTS = {
    "ru": {
        "native": [
            "Расскажи мне о столице России.",
            "Расскажи мне короткую историю.",
            "Какое самое популярное русское блюдо?",
            "Расскажи о футболе.",
            "Опиши Красную площадь.",
            "Что такое Масленица?",
            "Напиши короткое стихотворение.",
            "Каковы польза чтения?",
            "Расскажи о реке Волге.",
            "Какова история русской музыки?",
        ],
        "latin": [
            "Rasskazhi mne o stolitse Rossii.",
            "Rasskazhi mne korotkuyu istoriyu.",
            "Kakoye samoye populyarnoye russkoye blyudo?",
            "Rasskazhi o futbole.",
            "Opishi Krasnuyu ploshchad.",
            "Chto takoye Maslenitsa?",
            "Napishi korotkoye stikhotvoreniye.",
            "Kakovy polza chteniya?",
            "Rasskazhi o reke Volge.",
            "Kakova istoriya russkoy muzyki?",
        ],
    },
    "mr": {
        "native": [
            "भारताच्या राजधानीबद्दल काही सांगा.",
            "मला एक छोटी गोष्ट सांगा.",
            "सर्वोत्तम महाराष्ट्रीयन पदार्थ कोणता?",
            "क्रिकेटबद्दल सांगा.",
            "हिमालय पर्वताचे वर्णन करा.",
            "दिवाळी सण काय आहे?",
            "एक कविता लिहा.",
            "योगाचे फायदे सांगा.",
            "गंगा नदीबद्दल सांगा.",
            "भारतीय संगीताचा इतिहास काय आहे?",
        ],
        "latin": [
            "Bharatachya rajdhaanibaddal kaahi sanga.",
            "Mala ek chhoti gosht sanga.",
            "Sarvottam Maharashtriyan padarth konata?",
            "Cricket baddal sanga.",
            "Himalaya parvataache varnan kara.",
            "Divali san kaay aahe?",
            "Ek kavita liha.",
            "Yoga che fayde sanga.",
            "Ganga nadibaddal sanga.",
            "Bharatiya sangitaacha itihaas kaay aahe?",
        ],
    },
    "ur": {
        "native": [
            "پاکستان کے دارالحکومت کے بارے میں بتائیں۔",
            "مجھے ایک چھوٹی کہانی سنائیں۔",
            "سب سے اچھا پاکستانی کھانا کون سا ہے؟",
            "کرکٹ کے بارے میں بتائیں۔",
            "ہمالیہ پہاڑ کا ذکر کریں۔",
            "عید الفطر کیا ہے؟",
            "ایک نظم لکھیں۔",
            "پڑھنے کے فوائد بتائیں۔",
            "دریائے سندھ کے بارے میں بتائیں۔",
            "اردو موسیقی کی تاریخ کیا ہے؟",
        ],
        "latin": [
            "Pakistan ke darul-hukumat ke baare mein batayein.",
            "Mujhe ek chhoti kahani sunayein.",
            "Sab se achha Pakistani khana kaun sa hai?",
            "Cricket ke baare mein batayein.",
            "Himalaya pahaad ka zikar karein.",
            "Eid al-Fitr kya hai?",
            "Ek nazm likhein.",
            "Parhne ke fawaid batayein.",
            "Darya-e-Sindh ke baare mein batayein.",
            "Urdu mosiqi ki tareekh kya hai?",
        ],
    },
    "el": {
        "native": [
            "Πες μου για την πρωτεύουσα της Ελλάδας.",
            "Πες μου μια σύντομη ιστορία.",
            "Ποιο είναι το καλύτερο ελληνικό φαγητό;",
            "Μίλα μου για το ποδόσφαιρο.",
            "Περίγραψε την Ακρόπολη.",
            "Τι είναι το Πάσχα;",
            "Γράψε ένα σύντομο ποίημα.",
            "Ποια είναι τα οφέλη του διαβάσματος;",
            "Πες μου για τη θάλασσα του Αιγαίου.",
            "Ποια είναι η ιστορία της ελληνικής μουσικής;",
        ],
        "latin": [
            "Pes mou gia tin protevousa tis Elladas.",
            "Pes mou mia syntomi istoria.",
            "Poio einai to kalytero elliniko fagito?",
            "Mila mou gia to podosfairo.",
            "Perigrapse tin Akropoli.",
            "Ti einai to Pascha?",
            "Grapse ena syntomo poiima.",
            "Poia einai ta ofeli tou diavasmatos?",
            "Pes mou gia ti thalassa tou Aigaiou.",
            "Poia einai i istoria tis ellinikis mousikis?",
        ],
    },
    "ja": {
        "native": [
            "日本の首都について教えてください。",
            "短い物語を話してください。",
            "一番おいしい日本料理は何ですか？",
            "野球について話してください。",
            "富士山について説明してください。",
            "お正月とは何ですか？",
            "短い詩を書いてください。",
            "読書の利点を教えてください。",
            "利根川について教えてください。",
            "日本の音楽の歴史は何ですか？",
        ],
        "latin": [
            "Nihon no shuto ni tsuite oshiete kudasai.",
            "Mijikai monogatari wo hanashite kudasai.",
            "Ichiban oishii Nihon ryouri wa nan desu ka?",
            "Yakyuu ni tsuite hanashite kudasai.",
            "Fujisan ni tsuite setsumei shite kudasai.",
            "Oshougatsu to wa nan desu ka?",
            "Mijikai shi wo kaite kudasai.",
            "Dokusho no riten wo oshiete kudasai.",
            "Tonegawa ni tsuite oshiete kudasai.",
            "Nihon no ongaku no rekishi wa nan desu ka?",
        ],
    },
    "th": {
        "native": [
            "บอกเกี่ยวกับเมืองหลวงของประเทศไทย",
            "เล่าเรื่องสั้นให้ฟังหน่อย",
            "อาหารไทยที่อร่อยที่สุดคืออะไร",
            "เล่าเกี่ยวกับฟุตบอล",
            "อธิบายเกี่ยวกับภูเขาดอยอินทนนท์",
            "วันสงกรานต์คืออะไร",
            "เขียนบทกวีสั้นๆ ให้หน่อย",
            "ประโยชน์ของการอ่านหนังสือคืออะไร",
            "เล่าเกี่ยวกับแม่น้ำเจ้าพระยา",
            "ประวัติดนตรีไทยเป็นอย่างไร",
        ],
        "latin": [
            "Bok kiao kap mueang luang khong prathet Thai.",
            "Lao rueang san hai fang noi.",
            "Ahan Thai thi aroi thi sut khue arai?",
            "Lao kiao kap futbon.",
            "Athibai kiao kap phukhao Doi Inthanon.",
            "Wan Songkran khue arai?",
            "Khian botkawi san san hai noi.",
            "Prayot khong kan an nangsue khue arai?",
            "Lao kiao kap mae nam Chao Phraya.",
            "Prawat dontri Thai pen yang rai?",
        ],
    },
    "zh": {
        "native": [
            "请介绍一下中国的首都。",
            "请给我讲一个短故事。",
            "最好吃的中国菜是什么?",
            "请讲讲足球。",
            "请描述长城。",
            "春节是什么?",
            "请写一首短诗。",
            "读书的好处是什么?",
            "请介绍一下长江。",
            "中国音乐的历史是什么?",
        ],
        "latin": [
            "Qing jieshao yixia Zhongguo de shoudu.",
            "Qing gei wo jiang yige duan gushi.",
            "Zui haochi de Zhongguo cai shi shenme?",
            "Qing jiangjiang zuqiu.",
            "Qing miaoshu Changcheng.",
            "Chunjie shi shenme?",
            "Qing xie yi shou duan shi.",
            "Dushu de haochu shi shenme?",
            "Qing jieshao yixia Changjiang.",
            "Zhongguo yinyue de lishi shi shenme?",
        ],
    },
    "en_cy": {
        "native": [
            "Телл ми эбаут зэ кэпитэл оф Инглэнд.",
            "Телл ми э шорт стори.",
            "Уот из зэ бэст Инглиш диш?",
            "Телл ми эбаут футбол.",
            "Дискрайб Биг Бэн.",
            "Уот из Кристмэс?",
            "Райт э шорт поэм.",
            "Уот ар зэ бэнэфитс оф ридинг?",
            "Телл ми эбаут зэ Риврэ Темз.",
            "Уот из зэ хистэри оф Инглиш мюзик?",
        ],
        "latin": [
            "Tell me about the capital of England.",
            "Tell me a short story.",
            "What is the best English dish?",
            "Tell me about football.",
            "Describe Big Ben.",
            "What is Christmas?",
            "Write a short poem.",
            "What are the benefits of reading?",
            "Tell me about the River Thames.",
            "What is the history of English music?",
        ],
    },
    "es": {
        "native": [],
        "latin": [
            "Cuéntame sobre la capital de España.",
            "Cuéntame una historia corta.",
            "¿Cuál es el mejor plato español?",
            "Háblame del fútbol.",
            "Describe la Sagrada Familia.",
            "¿Qué es la Navidad?",
            "Escribe un poema corto.",
            "¿Cuáles son los beneficios de leer?",
            "Háblame del río Ebro.",
            "¿Cuál es la historia de la música española?",
        ],
    },
}


# ────────────────────────────────────────────────────────────────
# GlotLID
# ────────────────────────────────────────────────────────────────

class LanguageDetector:
    def __init__(self):
        import fasttext
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="cis-lmu/glotlid", filename="model.bin")
        self.model = fasttext.load_model(path)

    @staticmethod
    def _clean(text):
        return " ".join(text.split())

    def detect(self, text):
        """Returns (full_code, script_suffix)."""
        text = self._clean(text)
        if not text:
            return "empty", "empty"
        labels, _ = self.model.predict(text)
        full = labels[0].replace("__label__", "")
        script = full.split("_")[-1] if "_" in full else full
        return full, script


# ────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif", "mathtext.fontset": "cm",
    })
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "axes.linewidth": 1.0})


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    def default(o):
        if isinstance(o, np.ndarray):   return o.tolist()
        if isinstance(o, np.integer):   return int(o)
        if isinstance(o, np.floating):  return float(o)
        raise TypeError(f"Unserializable: {type(o)}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=default)


def get_model_layers(model, layers_attr):
    obj = model
    for attr in layers_attr:
        obj = getattr(obj, attr)
    return obj


def get_attn_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise RuntimeError("No attention submodule")


def in_native_range(char_code, ranges):
    for lo, hi in ranges:
        if lo <= char_code <= hi:
            return True
    return False


def script_ratios(text, native_ranges):
    nat = lat = 0
    for c in text:
        cp = ord(c)
        if in_native_range(cp, native_ranges):
            nat += 1
        elif "a" <= c.lower() <= "z":
            lat += 1
    total = nat + lat
    if total == 0:
        return 0.0, 0.0
    return nat / total, lat / total


def measure_text(text, native_ranges, native_script, detector):
    """Returns dict with both char-ratio and GlotLID indicators."""
    nr, lr = script_ratios(text, native_ranges)
    full, script = detector.detect(text)
    return {
        "text": text,
        "native_ratio": nr,
        "latin_ratio":  lr,
        "glotlid_full": full,
        "glotlid_script": script,
        "glotlid_is_native_script": (native_script is not None and script == native_script),
        "glotlid_is_latin_script":  (script == "Latn"),
    }


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


# ────────────────────────────────────────────────────────────────
# Activation caching and patching
# ────────────────────────────────────────────────────────────────

def cache_last_token_acts(wrapper, text):
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
        wrapper.model(**inputs)
    for h in handles:
        h.remove()
    return cache


def _format_prompt(wrapper, msg):
    if hasattr(wrapper.tokenizer, "apply_chat_template"):
        try:
            return wrapper.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            pass
    return msg


def _install_patch_hook(wrapper, interventions):
    handles = []
    for layer_idx, head_ops in interventions.items():
        attn = get_attn_module(wrapper.layers[layer_idx])
        def make_hook(ops):
            def pre_hook(mod, inp):
                x = inp[0].clone()
                x_h = x.view(x.shape[0], x.shape[1], wrapper.n_heads, wrapper.head_dim)
                for head_idx, repl in ops:
                    x_h[:, -1, head_idx, :] = repl.to(x_h.dtype).to(x_h.device)
                return (x_h.view(x.shape[0], x.shape[1], -1),) + inp[1:]
            return pre_hook
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(head_ops)))
    return handles


def generate(wrapper, text, max_new_tokens=60):
    formatted = _format_prompt(wrapper, text)
    inputs = wrapper.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    plen = inputs.input_ids.shape[1]
    with torch.no_grad():
        gen = wrapper.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=wrapper.tokenizer.pad_token_id,
        )
    return wrapper.tokenizer.decode(gen[0, plen:], skip_special_tokens=True).strip()


def generate_patched(wrapper, text, interventions, max_new_tokens=60):
    handles = _install_patch_hook(wrapper, interventions)
    try:
        return generate(wrapper, text, max_new_tokens)
    finally:
        for h in handles:
            h.remove()


# ────────────────────────────────────────────────────────────────
# Head-source loading
# ────────────────────────────────────────────────────────────────

def load_top_heads(model, lang, k):
    path = os.path.join(RESULTS_DIR, model, lang, "patch", "top.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["top_heads_by_abs_effect"][:k]
    return [(int(it["layer"]), int(it["head"])) for it in items]


# ────────────────────────────────────────────────────────────────
# Per-(target_lang, head_source) experiment
# ────────────────────────────────────────────────────────────────

def _summarize_pairs(pairs, native_script):
    """Compute char-ratio and GlotLID flip rates for both directions."""
    n2l_char = [1 if p["native_input_patch_latin_all"]["latin_ratio"] >= FLIP_THRESHOLD else 0
                for p in pairs]
    l2n_char = [1 if p["latin_input_patch_native_all"]["native_ratio"] >= FLIP_THRESHOLD else 0
                for p in pairs]
    n2l_glot = [1 if p["native_input_patch_latin_all"]["glotlid_is_latin_script"] else 0
                for p in pairs]
    l2n_glot = [1 if p["latin_input_patch_native_all"]["glotlid_is_native_script"] else 0
                for p in pairs]
    return {
        # char-ratio (original)
        "native_to_latin_flip_rate":      float(np.mean(n2l_char)),
        "latin_to_native_flip_rate":      float(np.mean(l2n_char)),
        "mean_flip_rate":                 float((np.mean(n2l_char) + np.mean(l2n_char)) / 2),
        # GlotLID script-suffix
        "native_to_latin_glotlid_rate":   float(np.mean(n2l_glot)),
        "latin_to_native_glotlid_rate":   float(np.mean(l2n_glot)),
        "mean_glotlid_rate":              float((np.mean(n2l_glot) + np.mean(l2n_glot)) / 2),
        "n_prompts":                      len(pairs),
    }


def run_one_language(wrapper, target_lang, head_sources, detector, max_new_tokens=60):
    cfg = LANG_CONFIG[target_lang]
    prompts = PROMPTS[target_lang]
    native_texts = prompts["native"]
    latin_texts  = prompts["latin"]
    native_ranges = cfg["native_ranges"]
    native_script = cfg["native_script"]

    is_control = len(native_texts) == 0
    if is_control:
        out = {"__note": "Latin-only control: no parallel pairs"}
        out["baselines"] = []
        for t in tqdm(latin_texts, desc=f"{target_lang} baselines"):
            txt = generate(wrapper, t, max_new_tokens)
            m = measure_text(txt, native_ranges, native_script, detector)
            out["baselines"].append({"prompt": t, **m})
        return out

    # Cache activations
    print(f"  Caching {target_lang} activations...")
    nat_caches = []
    lat_caches = []
    for t in tqdm(native_texts, desc=f"{target_lang} native cache", leave=False):
        nat_caches.append(cache_last_token_acts(wrapper, t))
    for t in tqdm(latin_texts, desc=f"{target_lang} latin cache", leave=False):
        lat_caches.append(cache_last_token_acts(wrapper, t))

    # Baselines
    print(f"  Baselines for {target_lang}...")
    base_nat = [generate(wrapper, t, max_new_tokens) for t in native_texts]
    base_lat = [generate(wrapper, t, max_new_tokens) for t in latin_texts]

    out = {"baselines": {
        "native_inputs": [
            {"prompt": native_texts[i],
             **measure_text(base_nat[i], native_ranges, native_script, detector)}
            for i in range(len(native_texts))
        ],
        "latin_inputs": [
            {"prompt": latin_texts[i],
             **measure_text(base_lat[i], native_ranges, native_script, detector)}
            for i in range(len(latin_texts))
        ],
    }}

    for src_key, heads in head_sources.items():
        per_layer = {}
        for l, h in heads:
            per_layer.setdefault(l, []).append(h)

        src_out = {"heads": [f"L{l}H{h}" for l, h in heads], "prompts": []}
        for i in tqdm(range(len(native_texts)), desc=f"{target_lang}/{src_key}", leave=False):
            patch_nil = {l: [(h, lat_caches[i][l][h]) for h in hs]
                         for l, hs in per_layer.items()}
            patch_lin = {l: [(h, nat_caches[i][l][h]) for h in hs]
                         for l, hs in per_layer.items()}

            nil_txt = generate_patched(wrapper, native_texts[i], patch_nil, max_new_tokens)
            lin_txt = generate_patched(wrapper, latin_texts[i],  patch_lin, max_new_tokens)

            src_out["prompts"].append({
                "native_prompt": native_texts[i],
                "latin_prompt":  latin_texts[i],
                "baseline_native": measure_text(base_nat[i], native_ranges, native_script, detector),
                "baseline_latin":  measure_text(base_lat[i], native_ranges, native_script, detector),
                "native_input_patch_latin_all": measure_text(nil_txt, native_ranges, native_script, detector),
                "latin_input_patch_native_all": measure_text(lin_txt, native_ranges, native_script, detector),
            })

        src_out["summary"] = _summarize_pairs(src_out["prompts"], native_script)
        out[src_key] = src_out

    return out


# ────────────────────────────────────────────────────────────────
# Plotting (no titles)
# ────────────────────────────────────────────────────────────────

def _plot_success_matrix(all_results, langs, sources, metric_key, cbar_label, out_path):
    set_style()
    mat = np.full((len(langs), len(sources)), np.nan)
    for i, lng in enumerate(langs):
        for j, src in enumerate(sources):
            if src in all_results[lng] and "summary" in all_results[lng][src]:
                mat[i, j] = all_results[lng][src]["summary"].get(metric_key, np.nan)

    lang_labels = [f"{LANG_CONFIG[l]['name']}\n({l})" for l in langs]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.85 * len(sources) + 1),
                                    max(3, 0.4 * len(langs) + 1)))
    sns.heatmap(mat, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=sources, yticklabels=lang_labels,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": cbar_label})
    ax.set_xlabel("Head source", fontweight="bold")
    ax.set_ylabel("Target language", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_success_matrices(all_results, out_dir, random_ks=(5, 10)):
    langs = [l for l in all_results if l != "__meta"]
    sources = ["hi_top5", "hi_top10", "ar_top5", "ar_top10"]
    sources += [f"random{k}" for k in random_ks]

    _plot_success_matrix(
        all_results, langs, sources,
        metric_key="mean_flip_rate",
        cbar_label="Mean char-ratio flip rate",
        out_path=os.path.join(out_dir, "success_matrix.pdf"),
    )
    _plot_success_matrix(
        all_results, langs, sources,
        metric_key="mean_glotlid_rate",
        cbar_label="Mean GlotLID script-flip rate",
        out_path=os.path.join(out_dir, "success_matrix_glotlid.pdf"),
    )


def _plot_direction_decomposition(all_results, langs, sources,
                                  n2l_key, l2n_key, out_path,
                                  pair_sources=None, highlight_head=None,
                                  base_count=None):
    set_style()
    if pair_sources and highlight_head:
        sources = sources + [s for s in pair_sources if highlight_head in s]
    if base_count is None:
        base_count = len(sources)

    ncols = min(3, len(langs))
    nrows = (len(langs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(3.8, 0.45 * len(sources)) * ncols, 2.8 * nrows),
                             squeeze=False, sharey=True)
    pal = sns.color_palette("tab10")

    for i, lng in enumerate(langs):
        ax = axes[i // ncols, i % ncols]
        x = np.arange(len(sources))
        n2l, l2n, n2l_err, l2n_err = [], [], [], []
        for src in sources:
            s = all_results[lng].get(src, {}).get("summary", {})
            n2l.append(s.get(n2l_key, np.nan))
            l2n.append(s.get(l2n_key, np.nan))
            n2l_err.append(s.get(f"{n2l_key}_sem", 0.0))
            l2n_err.append(s.get(f"{l2n_key}_sem", 0.0))
        w = 0.38
        ax.bar(x - w/2, n2l, w, yerr=n2l_err, color=pal[0], label="N2L",
               edgecolor="black", linewidth=0.5, capsize=2)
        ax.bar(x + w/2, l2n, w, yerr=l2n_err, color=pal[3], label="L2N",
               edgecolor="black", linewidth=0.5, capsize=2)
        ax.axhline(FLIP_THRESHOLD, color="black", lw=0.6, alpha=0.3, ls="--")

        if len(sources) > base_count:
            ax.axvline(base_count - 0.5, color="gray", lw=0.8, alpha=0.5, ls=":")

        def _short(src):
            if src in sources[:base_count]:
                return src
            if highlight_head and highlight_head in src:
                lang_tag = src.split("_pair", 1)[0]
                inner = src[src.find("[")+1:src.rfind("]")]
                parts = inner.split(",")
                others = [p for p in parts if p != highlight_head]
                return f"{lang_tag}+{others[0] if others else '?'}"
            return src

        ax.set_xticks(x); ax.set_xticklabels([_short(s) for s in sources],
                                             fontsize=6.5, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(LANG_CONFIG[lng]["name"], fontsize=9, fontweight="bold")
        if i % ncols == 0:
            ax.set_ylabel("Flip rate", fontweight="bold")
        if i == 0:
            ax.legend(loc="best", fontsize=6)

    for j in range(len(langs), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_direction_decompositions(all_results, out_dir, random_ks=(5, 10),
                                  pair_sources=None, highlight_head=None):
    langs = [l for l in all_results if l != "__meta"]
    base = ["hi_top5", "hi_top10", "ar_top5", "ar_top10"]
    base += [f"random{k}" for k in random_ks]

    _plot_direction_decomposition(
        all_results, langs, base,
        n2l_key="native_to_latin_flip_rate", l2n_key="latin_to_native_flip_rate",
        out_path=os.path.join(out_dir, "direction_decomposition.pdf"),
        pair_sources=pair_sources, highlight_head=highlight_head,
        base_count=len(base),
    )
    _plot_direction_decomposition(
        all_results, langs, base,
        n2l_key="native_to_latin_glotlid_rate", l2n_key="latin_to_native_glotlid_rate",
        out_path=os.path.join(out_dir, "direction_decomposition_glotlid.pdf"),
        pair_sources=pair_sources, highlight_head=highlight_head,
        base_count=len(base),
    )


# ────────────────────────────────────────────────────────────────
# Random-seed aggregation
# ────────────────────────────────────────────────────────────────

def aggregate_random_seeds(all_results, target_langs, random_ks, n_seeds):
    metric_keys = [
        "native_to_latin_flip_rate",     "latin_to_native_flip_rate",     "mean_flip_rate",
        "native_to_latin_glotlid_rate",  "latin_to_native_glotlid_rate",  "mean_glotlid_rate",
    ]
    for lng in target_langs:
        lng_res = all_results[lng]
        for k in random_ks:
            seed_keys = [f"random{k}_s{i}" for i in range(n_seeds)
                         if f"random{k}_s{i}" in lng_res
                         and "summary" in lng_res[f"random{k}_s{i}"]]
            if not seed_keys:
                continue
            agg = {"is_aggregated": True, "n_seeds": len(seed_keys),
                   "seeds_used": seed_keys, "summary": {}}
            for mk in metric_keys:
                vals = np.array([lng_res[sk]["summary"][mk] for sk in seed_keys])
                agg["summary"][mk] = float(vals.mean())
                agg["summary"][f"{mk}_sem"] = (
                    float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                )
            agg["summary"]["n_prompts"] = lng_res[seed_keys[0]]["summary"]["n_prompts"]
            lng_res[f"random{k}"] = agg


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_CONFIG, required=True)
    p.add_argument("--target-langs", nargs="+",
                   default=["ru", "mr", "ur", "el", "ja", "th"],
                   choices=list(LANG_CONFIG.keys()))
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--random-seeds", type=int, default=1)
    p.add_argument("--random-ks", nargs="+", type=int, default=[5, 10])
    p.add_argument("--run-pairs", action="store_true", default=True)
    p.add_argument("--no-pairs", dest="run_pairs", action="store_false")
    p.add_argument("--pair-skip-langs", nargs="+", default=["es"])
    p.add_argument("--pair-highlight-head", default="L20H18")
    args = p.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)

    print("Loading hi/ar head sources (top-5, top-10) ...")
    head_sources = {
        "hi_top5":  load_top_heads(args.model, "hi", 5),
        "hi_top10": load_top_heads(args.model, "hi", 10),
        "ar_top5":  load_top_heads(args.model, "ar", 5),
        "ar_top10": load_top_heads(args.model, "ar", 10),
    }
    for k, v in head_sources.items():
        print(f"  {k}: {[f'L{l}H{h}' for l,h in v]}")

    wrapper = ModelWrapper(args.model)

    print("Loading GlotLID...")
    detector = LanguageDetector()

    rng = np.random.default_rng(SEED)
    n_L, n_H = wrapper.n_layers, wrapper.n_heads
    total = n_L * n_H
    print(f"\nGenerating {args.random_seeds} random head sets per k ∈ {args.random_ks}")
    for k in args.random_ks:
        for seed_i in range(args.random_seeds):
            flat_idx = rng.choice(total, size=k, replace=False)
            heads = [(int(i // n_H), int(i % n_H)) for i in flat_idx]
            key = f"random{k}_s{seed_i}"
            head_sources[key] = heads
            print(f"  {key}: {[f'L{l}H{h}' for l,h in heads]}")

    pair_sources = []
    if args.run_pairs:
        print("\nConstructing pair combinations (C(5,2) within each source's top-5)")
        for src_lang in ("hi", "ar"):
            top5 = head_sources[f"{src_lang}_top5"]
            for a, b in combinations(range(5), 2):
                ha, hb = top5[a], top5[b]
                key = f"{src_lang}_pair[L{ha[0]}H{ha[1]},L{hb[0]}H{hb[1]}]"
                head_sources[key] = [ha, hb]
                pair_sources.append(key)
        print(f"  {len(pair_sources)} pair sources total")

    all_results = {"__meta": {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "head_sources": {k: [[l, h] for l, h in v] for k, v in head_sources.items()},
    }}

    for tlang in args.target_langs:
        print(f"\n=== Target language: {tlang} ({LANG_CONFIG[tlang]['name']}) ===")
        if tlang in args.pair_skip_langs and pair_sources:
            sources_for_lang = {k: v for k, v in head_sources.items()
                                if k not in set(pair_sources)}
            print(f"  (skipping {len(pair_sources)} pair sources)")
        else:
            sources_for_lang = head_sources
        all_results[tlang] = run_one_language(
            wrapper, tlang, sources_for_lang, detector,
            max_new_tokens=args.max_new_tokens,
        )

    out_results = os.path.join(RESULTS_DIR, args.model, "cross_lingual")
    out_plots   = os.path.join(PLOTS_DIR,   args.model, "cross_lingual")
    os.makedirs(out_results, exist_ok=True); os.makedirs(out_plots, exist_ok=True)

    save_json(all_results, os.path.join(out_results, "generations.json"))

    aggregate_random_seeds(all_results, args.target_langs, args.random_ks, args.random_seeds)
    save_json(all_results, os.path.join(out_results, "generations.json"))

    summary_sources = ["hi_top5", "hi_top10", "ar_top5", "ar_top10"]
    summary_sources += [f"random{k}" for k in args.random_ks]
    summary_sources += pair_sources if args.run_pairs else []
    summary = {
        "model": args.model,
        "langs": {
            lng: {
                "is_control": len(LANG_CONFIG[lng]["native_ranges"]) == 0
                              or len(PROMPTS[lng]["native"]) == 0,
                "sources": {
                    src: all_results[lng][src]["summary"]
                    for src in summary_sources if src in all_results[lng]
                    and "summary" in all_results[lng][src]
                }
            }
            for lng in args.target_langs
        },
    }
    save_json(summary, os.path.join(out_results, "summary.json"))

    plot_success_matrices(all_results, out_plots, random_ks=tuple(args.random_ks))
    plot_direction_decompositions(
        all_results, out_plots, random_ks=tuple(args.random_ks),
        pair_sources=pair_sources if args.run_pairs else None,
        highlight_head=args.pair_highlight_head,
    )

    # Console summary (both metrics)
    print(f"\n=== Summary ({args.model}) — char-ratio mean flip ===")
    header_sources = ["hi_top5", "hi_top10", "ar_top5", "ar_top10"] + \
                     [f"random{k}" for k in args.random_ks]
    print(f"  {'Target':<22s}  " + "  ".join(f"{s:<10s}" for s in header_sources))
    for lng in args.target_langs:
        row = [f"{LANG_CONFIG[lng]['name']:<22s}"]
        for src in header_sources:
            s = all_results[lng].get(src, {}).get("summary", {})
            v = s.get("mean_flip_rate", float("nan"))
            row.append(f"{v:<10.2f}")
        print("  " + "  ".join(row))

    print(f"\n=== Summary ({args.model}) — GlotLID mean script-flip ===")
    print(f"  {'Target':<22s}  " + "  ".join(f"{s:<10s}" for s in header_sources))
    for lng in args.target_langs:
        row = [f"{LANG_CONFIG[lng]['name']:<22s}"]
        for src in header_sources:
            s = all_results[lng].get(src, {}).get("summary", {})
            v = s.get("mean_glotlid_rate", float("nan"))
            row.append(f"{v:<10.2f}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()