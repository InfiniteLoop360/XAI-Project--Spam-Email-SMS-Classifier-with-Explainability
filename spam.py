# ==========================
# Spam Email Classification Project (Optimized for PyCharm)
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
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
    'RandomForest': RandomForestClassifier(n_estimators=30, random_state=2),  # reduced for speed
    'ExtraTrees': ExtraTreesClassifier(n_estimators=30, random_state=2)       # reduced for speed
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
# Save stacking pipeline as best model
with open('spam_pipeline.pkl', 'wb') as f:
    pickle.dump(stack_pipe, f)
print("\nStacking pipeline saved as 'spam_pipeline.pkl'")

# ---------------------------
# 12. Plots (All at the End)
# ---------------------------
# Target distribution
plt.figure(figsize=(6,6))
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct='%0.2f')
plt.title("Target Distribution")
plt.show()

# Characters histogram
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_characters'], color='blue', label='ham', alpha=0.6)
sns.histplot(df[df['target']==1]['num_characters'], color='red', label='spam', alpha=0.6)
plt.legend()
plt.title("Number of Characters Distribution")
plt.show()

# Words histogram
plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'], color='blue', label='ham', alpha=0.6)
sns.histplot(df[df['target']==1]['num_words'], color='red', label='spam', alpha=0.6)
plt.legend()
plt.title("Number of Words Distribution")
plt.show()

# Correlation heatmap
numeric_df = df[['num_characters','num_words','num_sentences','target']]
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ---------------------------
# 13. Inference Example
# ---------------------------
sample_text = "Congratulations! You won a $1000 gift card. Claim now."
sample_num_characters = len(sample_text)

sample_df = pd.DataFrame({
    'clean_text': [clean_and_tokenize(sample_text)],
    'num_characters': [sample_num_characters]
})

prediction = stack_pipe.predict(sample_df)
print("\nInference Example:")
print("Message:", sample_text)
print("Prediction:", "spam" if prediction[0]==1 else "ham")
