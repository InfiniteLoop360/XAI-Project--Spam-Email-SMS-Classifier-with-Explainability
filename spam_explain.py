# ==========================
# Spam Email Classification Project (Optimized for PyCharm + Explainability)
# ==========================

# ---------------------------
# 1. Import Libraries
# ---------------------------
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score
from lime.lime_text import LimeTextExplainer
import pickle

# ---------------------------
# 2. Download NLTK Resources
# ---------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------------------
# 3. Load Data
# ---------------------------
df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
df.rename(columns={'v1':'target','v2':'text'}, inplace=True)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.drop_duplicates(keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------------------------
# 4. Feature Engineering
# ---------------------------
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# ---------------------------
# 5. Text Preprocessing
# ---------------------------
stop = set(stopwords.words('english'))
wnl = WordNetLemmatizer()

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)           # remove emails
    text = re.sub(r'\d+', ' ', text)               # remove numbers
    tokens = nltk.word_tokenize(text)
    tokens = [wnl.lemmatize(t) for t in tokens if t.isalpha() and t not in stop]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_and_tokenize)

# ---------------------------
# 6. Train-Test Split
# ---------------------------
X = df[['clean_text','num_characters']]
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)

# ---------------------------
# 7. Preprocessing Pipeline
# ---------------------------
text_transformer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', text_transformer, 'clean_text'),
        ('num', MinMaxScaler(), ['num_characters'])
    ]
)

# ---------------------------
# 8. Multiple Classifiers
# ---------------------------
clfs = {
    'MultinomialNB': MultinomialNB(),
    'LogisticRegression': LogisticRegression(solver='liblinear', penalty='l2'),
    'SVC': SVC(kernel='sigmoid', gamma=1.0, probability=True),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=30, random_state=2),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=30, random_state=2)
}

results = []
for name, clf in clfs.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    results.append((name, acc, prec))
    print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}")

# ---------------------------
# 9. Voting Classifier (Soft Voting)
# ---------------------------
voting_clf = VotingClassifier(
    estimators=[
        ('svc', clfs['SVC']),
        ('nb', clfs['MultinomialNB']),
        ('et', clfs['ExtraTrees'])
    ],
    voting='soft'
)

voting_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

voting_pipe.fit(X_train, y_train)
y_pred_voting = voting_pipe.predict(X_test)
print("\nVoting Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Precision:", precision_score(y_test, y_pred_voting))

# ---------------------------
# 10. Stacking Classifier
# ---------------------------
stack_estimators = [
    ('svc', clfs['SVC']),
    ('nb', clfs['MultinomialNB']),
    ('et', clfs['ExtraTrees'])
]
stack_final_estimator = RandomForestClassifier(n_estimators=30, random_state=2)

stacking_clf = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=stack_final_estimator
)

stack_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

stack_pipe.fit(X_train, y_train)
y_pred_stack = stack_pipe.predict(X_test)
print("\nStacking Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("Precision:", precision_score(y_test, y_pred_stack))

# ---------------------------
# 11. Save Best Pipeline
# ---------------------------
with open('spam_pipeline.pkl', 'wb') as f:
    pickle.dump(stack_pipe, f)
print("\nStacking pipeline saved as 'spam_pipeline.pkl'")

# ---------------------------
# 12. Plots
# ---------------------------
plt.figure(figsize=(6,6))
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct='%0.2f')
plt.title("Target Distribution")
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'], color='blue', label='ham', alpha=0.6)
sns.histplot(df[df['target']==1]['num_characters'], color='red', label='spam', alpha=0.6)
plt.legend()
plt.title("Number of Characters Distribution")
plt.show()

plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'], color='blue', label='ham', alpha=0.6)
sns.histplot(df[df['target']==1]['num_words'], color='red', label='spam', alpha=0.6)
plt.legend()
plt.title("Number of Words Distribution")
plt.show()

numeric_df = df[['num_characters','num_words','num_sentences','target']]
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------
# 13. Inference + Explainability
# ---------------------------
# Load pipeline
with open('spam_pipeline.pkl', 'rb') as f:
    stack_pipe = pickle.load(f)

# LIME Explainer for human-readable explanation
explainer = LimeTextExplainer(class_names=['ham','spam'])

def explain_prediction(message):
    cleaned = clean_and_tokenize(message)
    sample_df = pd.DataFrame({'clean_text':[cleaned],'num_characters':[len(message)]})
    prediction = stack_pipe.predict(sample_df)[0]
    pred_label = "spam" if prediction==1 else "ham"

    # LIME explanation
    def predict_fn(texts):
        df_temp = pd.DataFrame({'clean_text':[clean_and_tokenize(t) for t in texts],
                                'num_characters':[len(t) for t in texts]})
        return stack_pipe.predict_proba(df_temp)

    exp = explainer.explain_instance(message, predict_fn, num_features=5)
    print("\nMessage:", message)
    print("Prediction:", pred_label)
    print("Top contributing words:")
    for feature, weight in exp.as_list():
        print(f"  {feature}: {weight:.4f}")


# ---------------------------
# 14. Human-friendly Explainability
# ---------------------------
def explain_prediction(model_pipeline, message, top_n=5):
    from lime.lime_text import LimeTextExplainer

    # Preprocess
    clean_msg = clean_and_tokenize(message)
    num_chars = len(message)
    sample_df = pd.DataFrame({'clean_text': [clean_msg], 'num_characters': [num_chars]})

    # Predict
    pred = model_pipeline.predict(sample_df)[0]
    pred_label = "spam" if pred == 1 else "ham"

    # Initialize LIME explainer
    class_names = ['ham', 'spam']
    explainer = LimeTextExplainer(class_names=class_names)

    # Explain prediction
    exp = explainer.explain_instance(
        clean_msg,
        classifier_fn=lambda x: model_pipeline.predict_proba(
            pd.DataFrame({'clean_text': x, 'num_characters': [len(m) for m in x]})),
        num_features=top_n
    )

    # Get top words contributing to prediction
    top_words = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top_words_only = [f"'{w[0]}'" for w in top_words if (pred == 1 and w[1] > 0) or (pred == 0 and w[1] < 0)]

    # Human-friendly sentence
    if top_words_only:
        reason = f"The message is classified as **{pred_label.upper()}** because it contains words like {', '.join(top_words_only)}."
    else:
        reason = f"The message is classified as **{pred_label.upper()}** based on overall content."

    return pred_label, reason


# ---------------------------
# 15. Explainability with Colored Words
# ---------------------------
def explain_with_color(model_pipeline, message, top_n=5):
    from lime.lime_text import LimeTextExplainer

    clean_msg = clean_and_tokenize(message)
    num_chars = len(message)
    sample_df = pd.DataFrame({'clean_text': [clean_msg], 'num_characters': [num_chars]})

    # Prediction
    proba = model_pipeline.predict_proba(sample_df)[0]
    pred = np.argmax(proba)
    pred_label = "spam" if pred == 1 else "ham"
    confidence = proba[pred]

    # Initialize LIME explainer
    class_names = ['ham', 'spam']
    explainer = LimeTextExplainer(class_names=class_names)

    # Explain prediction
    exp = explainer.explain_instance(
        clean_msg,
        classifier_fn=lambda x: model_pipeline.predict_proba(
            pd.DataFrame({'clean_text': x, 'num_characters': [len(m) for m in x]})),
        num_features=top_n
    )

    # Top contributing words
    top_words = dict(exp.as_list()[:top_n])

    # Colorize words
    colored_message = []
    for word in message.split():
        clean_word = re.sub(r'\W+', '', word).lower()
        if clean_word in top_words:
            weight = top_words[clean_word]
            if (pred == 1 and weight > 0) or (pred == 0 and weight < 0):
                colored_word = f"\033[91m{word}\033[0m"  # red
            else:
                colored_word = f"\033[92m{word}\033[0m"  # green
        else:
            colored_word = word
        colored_message.append(colored_word)

    # Confidence bar
    bar_length = 30
    filled_length = int(bar_length * confidence)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    # Human-readable explanation
    top_reasons = [f"'{w}'" for w, wt in top_words.items() if (pred == 1 and wt > 0) or (pred == 0 and wt < 0)]
    if top_reasons:
        reason = f"The message is classified as **{pred_label.upper()}** because of words like {', '.join(top_reasons)}."
    else:
        reason = f"The message is classified as **{pred_label.upper()}** based on overall content."

    # Print
    print("\n" + "="*60)
    print(f"Message: {' '.join(colored_message)}")
    print(f"Prediction: {pred_label.upper()}   Confidence: {confidence:.2f}")
    print(f"[{bar}]")
    print(f"Explanation: {reason}")
    print("="*60 + "\n")


# ---------------------------
# Example Usage
# ---------------------------
messages = [
    "Congratulations! You won a $1000 gift card. Claim now.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Click here to secure it.",
    "Don't forget to submit the assignment by tomorrow.",
    "Free entry in a weekly competition. Win a prize!"
]

print("\nHuman-friendly Explainability Examples:\n")
for msg in messages:
    pred_label, reason = explain_prediction(stack_pipe, msg)
    print(f"Message: {msg}")
    print(f"Prediction: {pred_label}")
    print(f"Explanation: {reason}\n")


for msg in messages:
    explain_with_color(stack_pipe, msg)