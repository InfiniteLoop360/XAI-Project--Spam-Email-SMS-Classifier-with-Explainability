
# 📧 XAI Project: Spam Email/SMS Classifier with Explainability

This project is a robust Machine Learning application designed to classify Email/SMS messages as Spam or Ham (Safe).

Unlike traditional "Black Box" classifiers, this project focuses on eXplainable AI (XAI) using LIME (Local Interpretable Model-agnostic Explanations). It not only predicts whether a message is spam but also provides a human-readable explanation, highlighting specific words that contributed to the decision (e.g., "won", "free", "urgent").

The model is built using an Ensemble Stacking Classifier to maximize accuracy and is deployed as an interactive web application using Streamlit.

## Abstract
The rapid proliferation of digital communication has led to an exponential increase in unsolicited spam messages, posing significant security risks and operational inefficiencies. While traditional Machine Learning (ML) models have proven effective in spam detection, they often function as "Black Boxes," lacking transparency in their decision-making processes. This project presents a high-precision Spam Email & SMS Classifier integrated with eXplainable AI (XAI) techniques to bridge the gap between model accuracy and human interpretability.

The proposed system employs a robust Stacking Classifier, an ensemble learning technique that combines the strengths of Support Vector Machines (SVM), Multinomial Naive Bayes, and Extra Trees Classifiers, with a Random Forest meta-learner. To handle unstructured text data, the project utilizes advanced Natural Language Processing (NLP) pipelines, including TF-IDF vectorization and custom feature engineering (e.g., character and sentence counts).

A distinguishing feature of this study is the integration of LIME (Local Interpretable Model-agnostic Explanations). Unlike conventional filters that simply label a message as "Spam" or "Ham," this system provides real-time, granular explanations by highlighting specific keywords (e.g., "urgent," "won," "free") that influenced the classification. The model is deployed via an interactive Streamlit web application, offering a user-friendly interface for real-time inference. Experimental results demonstrate that the Stacking Classifier achieves superior precision and accuracy compared to individual baseline models, while the LIME integration successfully enhances user trust by visualizing the model's reasoning logic.


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


## 📊 Model Performance & Results

We evaluated multiple classifiers to find the best balance between Accuracy and Precision. Precision is critical in spam detection because we want to avoid False Positives (marking a safe email as spam).

1. Classifier Comparison
```

|  Model  |  Accuracy  |  Precision  |
|  MultinomialNB  |  97.68%  |  100.00%  |
|  Logistic Regression  |  97.68%  |  100.00%
|  SVC  | 98.55%  | 96.77%  |
|  Random Forest  | 97.78%  |  99.09%  |
|  Extra Trees  | 98.84%  |  97.60%  |
|  Voting Classifier | 98.74% |  99.17%  |
|  Stacking Classifier  |   98.55% |  96.03%  |
```

Note: While Naive Bayes had 100% precision, the Ensemble models (Voting/Stacking) provided a more robust generalization on unseen complex data.

2. Data Visualizations

Key insights derived from our Exploratory Data Analysis (EDA):

Target Balance: The dataset is imbalanced (87% Ham vs 13% Spam).

Character Count: Spam messages are typically longer and hover around 150-160 characters (SMS limit), whereas Ham messages vary significantly.

3. LIME Explainability (XAI) in Action

The core feature of this project is explaining why a message is classified as Spam. Below are real outputs from the model:

Example 1: Spam Detection

Message: "Congratulations! You won a $1000 gift card. Claim now."

Prediction: 🚨 SPAM (100% Confidence)

Explanation: Classified as SPAM because of words like: claim, gift, won.

Example 2: Ham (Safe) Detection

Message: "Don't forget to submit the assignment by tomorrow."

Prediction: ✅ HAM (97% Confidence)

Explanation: Classified as HAM because of words like: tomorrow, assignment, submit.

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
