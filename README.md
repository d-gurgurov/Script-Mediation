# The Latin Substrate: How Language Models Represent and Mediate Script Choice

[![arXiv](https://img.shields.io/badge/arXiv-2605.31363-b31b1b.svg)](https://arxiv.org/abs/2605.31363)

This repository contains the code for **"The Latin Substrate: How Language Models Represent and Mediate Script Choice"**, to appear at **BlackBoxNLP @ EMNLP 2026**.

## Abstract

Many languages are written in multiple scripts, requiring large language models (LLMs) to generate equivalent linguistic content in distinct orthographic forms. While prior work suggests that LLMs route information through shared latent representations, how they internally mediate script variation remains poorly understood.

We study this question by first examining per-layer output distributions with the logit lens, which reveals consistent latent romanization during transliteration, and then through representational and mechanistic analyses of script generation. At the representational level, we show that scripts of the same language become increasingly separable across layers and that a simple linear steering direction can flip a model’s output script while largely maintaining semantic content. The vector generalizes asymmetrically to writing systems unseen during construction, flipping non-Latin output to Latin reliably, but mapping Latin output into varied non-Latin scripts. At the mechanistic level, we localize a small set of late-layer attention heads that causally mediate script choice. These heads transfer across unrelated languages and writing systems, suggesting that script routing is implemented by language-agnostic components. Across both analyses, we observe a consistent directional asymmetry: non-Latin output is produced by a compact, identifiable gate, while Latin-script output emerges from diffuse contributions across the network. Collectively, our findings hint that LLMs organize script variation around shared latent representations while exhibiting a privileged substrate toward Latin script.

## Code

The repository is organized into two sets of experiments:

* **Representational-level:** `0-probe.sh`, `1-vectors.sh`, `2-steer.sh`, and `3-generality.sh`
* **Mechanistic-level:** `A-localize.sh`, `B-validate.sh`, `C-compare.sh`, and `D-agnostic.sh`

The corresponding Python scripts implement representation analysis, steering, probing, generality, and mechanistic localization/validation experiments. See the individual scripts for details.

## Citation

```bibtex
@article{gurgurov2026latin,
  title={The Latin Substrate: How Language Models Represent and Mediate Script Choice},
  author={Gurgurov, Daniil and Saji, Alan and Trinley, Katharina and van Genabith, Josef and Ostermann, Simon},
  journal={arXiv preprint arXiv:2605.31363},
  year={2026}
}
```
