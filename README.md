# Fake News Generator
# Fake News Detector 

A machine learning project that detects whether a given news article is **real or fake** using **Natural Language Processing (NLP)** techniques. Built with Python, Scikit-learn, and trained using a Passive Aggressive Classifier.

---

## Features

- Classifies news as **FAKE** or **REAL**
- Uses **TfidfVectorizer** for text vectorization
- Trained using **Passive Aggressive Classifier**
- Easy to extend or integrate into web apps (e.g., Streamlit/Flask)
- Includes a **sample dataset** (`train.csv`) and trained model files

---


---

## 📊 Dataset

- **Columns**:
  - `title`: Title of the news article
  - `text`: Full text of the article
  - `subject`: Topic category (e.g., politics, world news)
  - `date`: Published date
  - `label`: **FAKE** or **REAL**

> Make sure your dataset is named `train.csv` and placed in the root directory.

---

## 🧠 How It Works

1. **Data Preprocessing**:
   - Extract `text` column for features
   - Use `label` as the target variable

2. **TF-IDF Vectorization**:
   - Transforms text into numeric vectors using `TfidfVectorizer`

3. **Model Training**:
   - Train a `PassiveAggressiveClassifier` on 80% of the data
   - Evaluate on the remaining 20%

4. **Model Saving**:
   - Save trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) using `joblib`

---


### Train the Model

```bash
python fake_news_detector.py

```

### How to Run
```bash
python app.py

---
```
##Output
![True](file:///C:/Users/sbhav/Downloads/Screenshot%202025-06-14%20203721.png)

![Fake News Detector Screenshot](file:///C:/Users/sbhav/Downloads/Screenshot%202025-06-14%20203451.png)





