# Argument Quality Assessment & Ranking

This project explores automated methods for evaluating the quality and persuasiveness of arguments using Natural Language Processing (NLP). The goal is to determine which of two arguments (A vs. B) is more convincing.

## Overview

The research investigates multiple approaches, ranging from metric-based encoding to Large Language Models (LLMs).

**Key Findings:**
- **Cross-Encoders** significantly outperformed other methods, achieving **~76.2% accuracy**.
- **LLMs (GPT-4o)** in a zero-shot setting achieved **~62% accuracy**, showing that domain-specific fine-tuning still beats general-purpose models for this specific task.
- **SBERT (Bi-Encoders)** with simple distance metrics performed near random chance (~50%), highlighting the need for full attention mechanisms in argumentation tasks.

## Datasets

1.  **UKP Convincing Arguments**: 11,650 labeled pairs.
2.  **ChangeMyView (CMV)**: 10,302 pairs derived from Reddit threads (Delta vs. Non-Delta comments).

## Approaches & Results

| Model | Architecture | Accuracy | F1 Score |
|-------|--------------|----------|----------|
| **Cross-Encoder (Best)**| `ms-marco-MiniLM-L-12-v2` (Fine-tuned) | **0.7622** | **0.76** |
| GPT-3.5 Turbo | Zero-Shot Prompting | 0.6400 | 0.59 |
| GPT-4o | Zero-Shot Prompting | 0.6200 | 0.50 |
| SBERT + Logistic Reg. | `paraphrase-MiniLM-L6-v2` | ~0.5000 | ~0.50 |

## Files

- `model.ipynb`: Primary research notebook containing data loading, preprocessing, model training, and evaluation.
- `model.py`: Script version of the logic (if applicable).
- `best_model_v2`: Directory containing the saved weights for the best-performing Cross-Encoder.

## References

- [UKP Convincing Arguments Corpus](https://github.com/UKPLab/acl2016-convincing-arguments)
- [ChangeMyView Dataset](https://arxiv.org/abs/1602.01103)