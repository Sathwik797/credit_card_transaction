import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("transactions.csv")

# Prepare data
X = df["message"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a text classification model
model = make_pipeline(CountVectorizer(), MultinomialNB())


# Train the model
model.fit(X_train, y_train)

# Save the trained model
with open("spam_detector.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

print("Model trained and saved successfully.")