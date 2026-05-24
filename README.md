# 🛡️ SpamGuard — Spam Email Classifier

A machine learning web app that classifies emails as **spam or legitimate (ham)** using a Naive Bayes classifier trained on TF-IDF features.

## ✨ Features

- **Real ML pipeline** — scikit-learn TF-IDF vectorizer + Multinomial Naive Bayes
- **Probability scores** — shows confidence for both spam and ham classes
- **TF-IDF feature visualization** — see which words most influenced the prediction
- **Model evaluation metrics** — accuracy, precision, recall, F1 score
- **Confusion matrix** — computed on a real held-out test set
- **Sample emails** — built-in spam and ham examples to try instantly
- **Dark-themed UI** — clean, professional interface

## 🚀 Getting Started

### 1. Clone / Download the project

```bash
cd spam_classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

## 🧠 How It Works

1. **Text preprocessing** — raw email text is cleaned and tokenized
2. **TF-IDF vectorization** — converts text to numerical features (1-gram + 2-gram, top 5,000 features)
3. **Naive Bayes classification** — Multinomial NB model predicts spam probability
4. **Evaluation** — model is evaluated on a 25% held-out test set, metrics shown in sidebar

## 📊 Model Details

| Component | Details |
|-----------|---------|
| Algorithm | Multinomial Naive Bayes |
| Vectorizer | TF-IDF (unigram + bigram) |
| Max features | 5,000 |
| Train/test split | 75% / 25% |
| Smoothing (α) | 0.1 |

## 🗂️ Project Structure

```
spam_classifier/
├── app.py               # Flask backend + ML model
├── templates/
│   └── index.html       # Frontend UI
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🛠️ Tech Stack

- **Backend** — Python, Flask, scikit-learn
- **ML** — TF-IDF, Multinomial Naive Bayes
- **Frontend** — HTML, CSS, Vanilla JS
