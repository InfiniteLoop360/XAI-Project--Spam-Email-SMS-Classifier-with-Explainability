
# 📧 XAI Project: Spam Email/SMS Classifier with Explainability

This project is a robust Machine Learning application designed to classify Email/SMS messages as Spam or Ham (Safe).

Unlike traditional "Black Box" classifiers, this project focuses on eXplainable AI (XAI) using LIME (Local Interpretable Model-agnostic Explanations). It not only predicts whether a message is spam but also provides a human-readable explanation, highlighting specific words that contributed to the decision (e.g., "won", "free", "urgent").

The model is built using an Ensemble Stacking Classifier to maximize accuracy and is deployed as an interactive web application using Streamlit.


## Key Features

- Advanced Text Preprocessing: Includes cleaning, tokenization, and lemmatization.

- Feature Engineering: Extracts meta-features like message length, word count, and sentence count.

- Ensemble Learning: Uses a Stacking Classifier combining Support Vector Machine (SVM), Naive Bayes, and Extra Trees.

- Explainability (XAI): Integrates LIME to visualize why a prediction was made, color-coding words based on their spam/ham probability.

- Interactive UI: A user-friendly Streamlit web app for real-time inference.

## Tech Stack

**Language:** Python

**Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn

**Explainability:** LIME

**Deployment:** Streamlit

**Serialization:** Pickle


## 📖 Step-by-Step Implementation Guide

Here is a detailed breakdown of how this project was constructed, corresponding to the logic in spam_explain.py and spam_app.py.

- Phase 1: Data Processing & Engineering

- Data Loading:

The project uses the SMS Spam Collection dataset (spam.csv).

Columns are renamed for clarity (v1 -> target, v2 -> text), and duplicates are removed to prevent data leakage.

- Label Encoding:

The target labels are converted into numerical format: Ham = 0, Spam = 1.

- Feature Engineering:

Before cleaning the text, we extract structural features which are strong indicators of spam:

num_characters: Total length of the message.

num_words: Total word count.

num_sentences: Total sentence count.

Text Preprocessing (NLP Pipeline):

Cleaning: Regex is used to remove URLs, email addresses, and special characters.

Tokenization: Breaking sentences into individual words.

Stopword Removal: Removing common words (is, the, a) that add no semantic value.

Lemmatization: Converting words to their root form (e.g., "running" -> "run") using WordNetLemmatizer.

- Phase 2: Model Building & Training

- Data Transformation:

TF-IDF (Term Frequency-Inverse Document Frequency): Converts text data into numerical vectors (max 3000 features), capturing the importance of words relative to the dataset.

MinMax Scaling: Scales the num_characters feature to a range of 0-1.

ColumnTransformer: Combines both text vectors and numerical features into a single input array.

- Model Selection:

Several algorithms were tested, including Naive Bayes, Logistic Regression, SVC, Decision Trees, and Random Forests.

Metric: Precision was prioritized over accuracy to minimize False Positives (marking a safe email as spam).

Ensemble Learning (Stacking):

To achieve the best performance, a Stacking Classifier was implemented.

Base Learners: SVC, MultinomialNB, ExtraTrees.

Final Estimator: Random Forest.

Logic: The base learners make predictions, and the final Random Forest uses those predictions as inputs to make the final decision.

- Phase 3: Explainability (The XAI Component)

- LIME Integration:

I used LimeTextExplainer.

How it works: It perturbs (slightly changes) the input text to see how the model's prediction changes.

Visualization: It assigns weights to specific words.

Positive weight (+): Contributes to Spam.

Negative weight (-): Contributes to Ham.

Phase 4: Deployment

Streamlit Application (spam_app.py):

The trained pipeline is loaded using pickle.

A custom HTML generation function takes the LIME output and creates a visual representation where "Spammy" words are highlighted red and "Safe" words are highlighted green.

The user sees the prediction, the confidence score, and the reasoning.


##  📊 Results


The Stacking Classifier achieved high precision and accuracy, outperforming individual models.

- Accuracy: ~97-98%

- Precision: ~99% (Crucial for spam filters)


## How to Run Locally

Clone the project

```bash
  git clone https://github.com/InfiniteLoop360/XAI-Project--Spam-Email-SMS-Classifier-with-Explainability.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install numpy pandas nltk scikit-learn matplotlib seaborn lime streamlit
```
Train the Model (Optional):
If you want to retrain the model from scratch:

```bash
 python spam_explain.py
```
This will generate the spam_pipeline.pkl file.


Run the App:

```bash
  streamlit run spam_app.py
```


## Future Scope
- Deep Learning: Implementing LSTM or BERT for potentially better context understanding.

- Real-time API: Wrapping the model in a Flask/FastAPI backend for integration with email clients.

- User Feedback Loop: Allowing users to flag incorrect classifications to retrain the model dynamically.
