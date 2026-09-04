from datasets import load_dataset

langs = ["hin_Deva", "arb_Arab", "arb_Latn"]

for lang in langs:
    ds = load_dataset("facebook/flores", lang, split="devtest")
    
    output_file = f"flores_{lang}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in ds:
            f.write(example["sentence"] + "\n")
    
    print(f"Saved {lang} to {output_file}")