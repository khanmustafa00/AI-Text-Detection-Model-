# AI-Text-Detection-Model-
This study compares a Decision Tree (DT) classifier and a BERT-MLP pipeline for AI text detection. Evaluated on the 170,000-sample MAGE benchmark, the BERT-MLP achieved 91.78% accuracy, outperforming the DT (79.01% accuracy).

# AI Text Detection: A Comparative Study

This repository contains the code and methodology for a comparative study on detecting AI-generated text, conducted as part of the "Introduction to Artificial Intelligence" course (IBA Karachi, Spring 2026).

## Overview
As Large Language Models (LLMs) become more prevalent, maintaining academic integrity has become increasingly challenging. This project explores binary classification approaches to distinguish between human-written and AI-generated text.

## Models
1. **Decision Tree (DT):** A custom, from-scratch implementation operating on 16 stylometric features. It offers high interpretability, crucial for explaining AI-authorship accusations in academic settings.
2. **BERT-MLP:** A Multi-Layer Perceptron operating on 80-dimensional BERT embeddings (compressed via PCA). It captures semantic coherence and discourse structure beyond surface-level patterns.

## Key Results
| Metric | Decision Tree | BERT-MLP |
| :--- | :--- | :--- |
| Test Accuracy | 79.01% | 91.78% |
| ROC-AUC | 0.8882 | 0.9751 |

## Ethical Considerations
To prioritize academic integrity and minimize false-positive accusations against students, we have implemented a configurable inference threshold (defaulting to 0.70), balancing the trade-off between false positives and false negatives.


