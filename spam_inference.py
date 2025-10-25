import pickle
import pandas as pd

# Load the saved pipeline
with open('spam_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# List of messages to classify
messages = [
    "Congratulations! You won a $1000 gift card. Claim now.",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Click here to secure it.",
    "Don't forget to submit the assignment by tomorrow.",
    "Free entry in a weekly competition. Win a prize!"
]

# Convert to DataFrame like the pipeline expects
df_messages = pd.DataFrame({
    'clean_text': messages,
    'num_characters': [len(msg) for msg in messages]
})

# Make predictions
predictions = pipeline.predict(df_messages)

# Display results
for msg, pred in zip(messages, predictions):
    label = "spam" if pred == 1 else "ham"
    print(f"Message: {msg}\nPrediction: {label}\n")
