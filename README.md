# Mental Health Signal Detection from Social Media

Text classification system to detect mental health indicators in Reddit posts. Classifies content into five categories: stress, depression, bipolar disorder, personality disorder, and anxiety.

## Results

| Model | Macro F1 Score | Status |
|-------|---------------|--------|
| BERT Fine-tuned | 81.06% | ✅ Target achieved |
| Logistic Regression | 78.12% | Baseline |
| Naive Bayes | 75.08% | Baseline |

Target was 80% F1. BERT model was trained on Google Colab with GPU.

## Tech Stack

- Python 3.13
- PyTorch + HuggingFace Transformers (BERT)
- scikit-learn (Logistic Regression, Naive Bayes)
- NLTK, spaCy (text preprocessing)
- Streamlit (dashboard)
- Google Colab (model training)

## Dataset

Reddit Mental Health Dataset from Kaggle (~5,957 posts)

- Source: https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data
- Classes: `stress`, `depression`, `bipolar`, `personality_disorder`, `anxiety`

> **Note:** Dataset and trained model files are excluded from this repo due to size. See setup instructions below.

## Setup

1. Clone the repo
2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. Download dataset:
   ```bash
   python data_downloader.py
   ```
   Or manually download from Kaggle and place in `data/raw/`

5. Download BERT model (optional):
   - Download `bert_mental_health.zip` from [your link]
   - Extract to `models/bert_mental_health/`
   - Or train on Colab using `notebooks/bert_finetune_colab.ipynb`

## Running the Project

Preprocessing:
```bash
python src/preprocessing.py
```

Feature extraction:
```bash
python src/features.py
```

Train/test split:
```bash
python src/split_data.py
```

Baseline models:
```bash
python src/baseline_models.py
```

Dashboard:
```bash
streamlit run app/app_custom.py
```

## Project Structure

```
mental_health_nlp/
├── app/                    # Streamlit dashboard
├── data/
│   ├── raw/               # Original dataset (not tracked)
│   └── processed/         # Cleaned data (not tracked)
├── models/                # Trained models (not tracked)
├── notebooks/             # Colab notebooks
├── src/                   # Source code
│   ├── preprocessing.py
│   ├── features.py
│   ├── split_data.py
│   ├── baseline_models.py
│   ├── bert_classifier.py
│   └── evaluate_all.py
├── reports/               # Figures and results
├── data_downloader.py     # Dataset download script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Notes

- Large files (dataset, model weights) are gitignored
- BERT training was done on Google Colab GPU due to local hardware constraints
- Fine-tuned BERT model files should be downloaded separately or trained via the provided notebook
