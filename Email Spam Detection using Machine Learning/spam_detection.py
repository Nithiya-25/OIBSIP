import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Fix columns if needed
if 'v1' in data.columns and 'v2' in data.columns:
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Sample email prediction
sample_email = ["Congratulations! You have won a free gift card. Click now"]
sample_vec = vectorizer.transform(sample_email)
prediction = model.predict(sample_vec)
print("\nSample Email Prediction:")
print("Spam" if prediction[0] == 1 else "Not Spam")

# User input prediction
print("\n=== Email Spam Detection (User Input) ===")
user_email = input("Enter an email/message to check: ")
user_vec = vectorizer.transform([user_email])
user_prediction = model.predict(user_vec)

if user_prediction[0] == 1:
    print("🚨 Result: SPAM EMAIL")
else:
    print("✅ Result: NOT SPAM")