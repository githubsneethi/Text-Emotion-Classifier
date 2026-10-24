# 🎭 Text Emotion Classifier

An end-to-end Machine Learning and Streamlit application that classifies the underlying emotion of any input English text into one of six categories: **Sadness (0)**, **Joy (1)**, **Love (2)**, **Anger (3)**, **Fear (4)**, and **Surprise (5)**.

This project leverages a large dataset containing over 416,000 text samples to compare classical Machine Learning approaches (Logistic Regression, Linear Support Vector Machines, Random Forests) to achieve a considerably decent accuracy and a deployable solution.

---

## 🚀 Live Demo & Deployment
The application is designed to run via a Streamlit dashboard. 

### ⚡ Quick Preview (via Streamlit)
You can view this on know-your-emotions.streamlit.app and test it. Thanks, Streamlit, for helping us deploy the app!

## 📊 Dataset
The model is trained on the popular **Emotions Dataset** by *Nelgiriyewithana* via Kaggle. It contains highly unbalanced categories:
- **Joy** (~141k samples)
- **Sadness** (~121k samples)
- **Anger** (~57k samples)
- **Fear** (~47k samples)
- **Love** (~34k samples)
- **Surprise** (~15k samples)

---

## 🛠️ Model Architecture & Evaluation

We benchmarked multiple representation techniques and classifiers:

1. **TF-IDF + Logistic Regression**
   - Unigrams vs Bigrams with `sublinear_tf` and custom class weighting.
   - **Accuracy:** ~90-91%
2. **TF-IDF + Linear SVC (Best Performing Classical Model)**
   - Achieved solid macro performance with balanced weights.
   - **Accuracy:** ~90% (with exceptional speed and small disk footprint)
3. **Sentence Transformers (all-MiniLM-L6-v2) + Logistic Regression**
   - Evaluated dense semantic embeddings.
   - **Accuracy:** ~65%
4. **TF-IDF + Random Forest Classifier**
   - Constrained to prevent unconstrained tree size.
   - **Accuracy:** ~81%

Our chosen production pipeline employs the **Linear Support Vector Classifier (LinearSVC)** paired with a **tuned Bigram TF-IDF Vectorizer** due to its outstanding generalization and lightweight inference footprint.

---

## 📂 Repository Structure

```directory
├── app.py                  # Streamlit web application script
├── emotion_model.pkl       # Saved trained LinearSVC classifier
├── tfidf_vectorizer.pkl    # Saved fitted TF-IDF Bigram Vectorizer
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation & Local Setup

Follow these steps to run the Streamlit app on your local machine:

### 1. Clone the repository
```bash
git clone https://github.com/githubsneethi/Text-Emotion-Classifier.git
cd Text-Emotion-Classifier
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
Ensure you have python installed, then run:
```bash
pip install streamlit scikit-learn numpy pandas joblib
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Your browser should automatically pop open a new window to `http://localhost:8501` showcasing the interface.

---

## 🧪 Streamlit App Features
- **Text input area**: Write any phrase or paragraph of your choice.
- **Dynamic Emotion prediction**: Uses the cached serialized model pipeline instantly.
- **Decision-function Softmax**: Outputs confidence probabilities for each class (even with LinearSVC).
- **Clean Bar Charts**: Sorted breakdowns of the exact classification confidence percentages across all emotions.

---

## 🤝 Contribution
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or open a Pull Request.

## 📄 License
This project is licensed under the MIT License.
```
