import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple dataset (no need to download anything)
data = {
    'text': [
        "Win money now", 
        "Free lottery offer", 
        "Call me later", 
        "Let's meet tomorrow",
        "Congratulations you won prize",
        "Important meeting today"
    ],
    'label': [1,1,0,0,1,0]
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model files created!")