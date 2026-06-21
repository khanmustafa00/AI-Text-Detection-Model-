# AI Text Detection: A Comparative Study

Comparing a Decision Tree (DT) and a BERT-MLP pipeline for detecting AI-generated text. Built for *Introduction to Artificial Intelligence* (IBA Karachi, Spring 2026), evaluated on the 170K-sample MAGE benchmark.

## Results
| Metric | Decision Tree | BERT-MLP |
| :--- | :--- | :--- |
| Accuracy | 79.01% | **91.78%** |
| ROC-AUC | 0.8882 | **0.9751** |

- **DT** — from-scratch, 16 stylometric features, fully interpretable.
- **BERT-MLP** — `bert-base-uncased` embeddings → PCA (768→80) → MLP (256→128→64, ReLU).

## Pipeline
| Script | Purpose |
| :--- | :--- |
| `bert_extract.py` | Loads MAGE, balances classes, extracts BERT embeddings, fits PCA → `extracted_bert_combined.csv` + `bert_pca.pkl` |
| `train_mlp.py` | Trains the MLP classifier, runs 5-fold CV, saves model/scaler/encoder + `mlp_training_stats.json` |
| `infer_mlp.py` | Interactive CLI — paste text, get a prediction |
| `ann_gui.py` | Streamlit GUI for the same pipeline (`streamlit run ann_gui.py`) |

**Required artifacts for inference:** `bert_pca.pkl`, `mlp_scaler.pkl`, `mlp_model.pkl`, `mlp_label_encoder.pkl`

## Project Structure
```
.
├── bert_extract.py              # MAGE loading + BERT embedding + PCA extraction
├── train_mlp.py                 # MLP training, evaluation, 5-fold CV
├── infer_mlp.py                 # CLI inference loop
├── ann_gui.py                   # Streamlit GUI
├── bert_pca.pkl                 # Fitted PCA (768→80) — from bert_extract.py
├── mlp_model.pkl                # Trained MLPClassifier — from train_mlp.py
├── mlp_scaler.pkl                # StandardScaler — from train_mlp.py
├── mlp_label_encoder.pkl          # LabelEncoder — from train_mlp.py
├── mlp_training_stats.json         # Train/val/test metrics, CV, loss curve
└── extracted_bert_combined.csv      # BERT-PCA features + labels (intermediate)
```

## Requirements
- Python 3.9+
- `torch`, `transformers` (BERT)
- `datasets` (loads MAGE from Hugging Face)
- `scikit-learn` (PCA, MLP, scaler, metrics)
- `pandas`, `numpy`, `joblib`
- `streamlit`, `matplotlib` (GUI only)

```bash
pip install torch transformers datasets scikit-learn pandas numpy joblib streamlit matplotlib
```

## Usage
```bash
# 1. Extract BERT+PCA features from MAGE (produces bert_pca.pkl + CSV)
python bert_extract.py

# 2. Train the MLP classifier (produces mlp_model.pkl, scaler, encoder, stats)
python train_mlp.py

# 3a. Run inference from the command line
python infer_mlp.py

# 3b. Or launch the GUI
streamlit run ann_gui.py
```

## Model Details
- **Base encoder:** `bert-base-uncased` (768-dim), max sequence length 512 tokens
- **Dimensionality reduction:** PCA, 768 → 80 components
- **Classifier:** `MLPClassifier` (scikit-learn), hidden layers (256, 128, 64), ReLU activation, Adam optimizer, early stopping
- **Dataset:** [MAGE](https://huggingface.co/datasets/yaful/MAGE) ([paper](https://arxiv.org/abs/2305.13242)), 170,000 samples, balanced human/machine-generated, ≥150 words/sample
- **Split:** 70% train / 15% val / 15% test, stratified

## Note
Inference threshold defaults to 0.70 to reduce false-positive AI-authorship accusations in academic settings.

## License
MIT License — see `LICENSE` for full text. Free to use, modify, and distribute with attribution. Provided "as is," without warranty of any kind.

## Citation
If you use this work, please cite the MAGE dataset/paper linked above alongside this repository.
