import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return ' '.join(ps.stem(w) for w in words if w not in stop_words)

# Load data from folder
import pandas as pd

true_df = pd.read_csv(r"D:\Cyber\fakenewsdetection\fakenewsdetection\News _dataset\True.csv")
fake_df = pd.read_csv(r"D:\Cyber\fakenewsdetection\fakenewsdetection\News _dataset\Fake.csv")

print("✅ CSVs loaded successfully")
print("True shape:", true_df.shape)
print("Fake shape:", fake_df.shape)



true_df['label'] = 0  # Real
fake_df['label'] = 1  # Fake

# Combine and preprocess
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df[['title', 'text', 'label']].dropna()
df['content'] = (df['title'] + " " + df['text']).apply(preprocess)

# Features and Labels
X = df['content']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")
