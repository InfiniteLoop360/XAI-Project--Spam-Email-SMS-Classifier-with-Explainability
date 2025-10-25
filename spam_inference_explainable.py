import pickle
import pandas as pd
import shap

# Load the saved pipeline
with open('spam_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Extract classifier and vectorizer for SHAP
model = pipeline.named_steps['clf']
vectorizer = pipeline.named_steps['preprocessor'].named_transformers_['tfidf']


# Function to explain prediction
def explain_message(message):
    df_message = pd.DataFrame({
        'clean_text': [message],
        'num_characters': [len(message)]
    })

    # Prediction
    pred = pipeline.predict(df_message)[0]
    pred_proba = pipeline.predict_proba(df_message)[0][pred]

    label = "spam" if pred == 1 else "ham"
    print(f"\nMessage: {message}")
    print(f"Prediction: {label} (Confidence: {pred_proba * 100:.2f}%)")

    # SHAP explanation
    X_transformed = vectorizer.transform(df_message['clean_text'])
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)

    print("\nWords influencing the decision:")
    shap.plots.text(shap_values[0])  # Visualize contribution of each word


# Interactive loop
print("=== Spam/Ham Classifier with Explainability ===")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nEnter message: ")
    if user_input.lower() == 'exit':
        break
    explain_message(user_input)
