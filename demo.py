#!/usr/bin/env python3
"""
Demo script for Fake News Detection Application
Shows sample predictions and features
"""

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
try:
    nltk.download('stopwords', quiet=True)
except:
    pass

def load_model():
    """Load the trained model and vectorizer"""
    try:
        with open('fake_news_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        print("❌ Model files not found. Please run train_model.py first.")
        return None, None

def preprocess(text):
    """Preprocess text for prediction"""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return ' '.join(ps.stem(w) for w in words if w not in stop_words)

def predict_news(model, vectorizer, title, content):
    """Predict if news is fake or true"""
    combined = title + " " + content
    processed = preprocess(combined)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0]
    
    return {
        "prediction": "TRUE" if pred == 0 else "FAKE",
        "confidence": float(max(confidence)),
        "true_probability": float(confidence[0]),
        "fake_probability": float(confidence[1])
    }

def main():
    """Main demo function"""
    print("🛡️ Fake News Detection Demo")
    print("=" * 40)
    
    # Load model
    model, vectorizer = load_model()
    if model is None:
        return
    
    print("✅ Model loaded successfully")
    print()
    
    # Sample news articles for demonstration
    sample_articles = [
        {
            "title": "Scientists Discover New Species of Deep Sea Creatures",
            "content": "A team of marine biologists has discovered several new species of deep sea creatures in the Pacific Ocean. The discovery was made during a research expedition using advanced underwater drones. The findings have been published in the prestigious Journal of Marine Biology and are expected to provide new insights into deep sea ecosystems.",
            "expected": "TRUE"
        },
        {
            "title": "BREAKING: Aliens Contact Earth Government",
            "content": "Incredible news! Aliens have made contact with world leaders and are sharing advanced technology. This is absolutely real and not fake at all. The government is hiding this from us but we have inside sources. They have been communicating for months and are planning to reveal themselves soon.",
            "expected": "FAKE"
        },
        {
            "title": "NASA Announces New Mars Mission for 2025",
            "content": "NASA has announced plans for a new Mars mission scheduled for 2025. The mission will focus on searching for signs of ancient microbial life and collecting samples for return to Earth. This represents a significant step forward in our understanding of the Red Planet and could provide crucial evidence about the possibility of life beyond Earth.",
            "expected": "TRUE"
        },
        {
            "title": "Miracle Cure Found for All Diseases",
            "content": "A revolutionary new treatment has been discovered that cures all diseases instantly! Doctors are shocked by this breakthrough. The treatment involves a simple pill that works for everything from cancer to the common cold. Big Pharma is trying to suppress this information but we have the truth!",
            "expected": "FAKE"
        },
        {
            "title": "Global Climate Summit Reaches Historic Agreement",
            "content": "World leaders at the COP28 climate summit have reached a historic agreement to accelerate the transition to renewable energy and reduce greenhouse gas emissions. The agreement includes commitments from major economies to achieve net-zero emissions by 2050 and provides funding for developing nations to adapt to climate change.",
            "expected": "TRUE"
        }
    ]
    
    print("📰 Testing Sample News Articles")
    print("-" * 40)
    
    correct_predictions = 0
    total_articles = len(sample_articles)
    
    for i, article in enumerate(sample_articles, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Expected: {article['expected']}")
        
        # Make prediction
        result = predict_news(model, vectorizer, article['title'], article['content'])
        
        # Display results
        prediction_icon = "🟢" if result['prediction'] == "TRUE" else "🔴"
        print(f"   Predicted: {prediction_icon} {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   True Probability: {result['true_probability']:.2%}")
        print(f"   Fake Probability: {result['fake_probability']:.2%}")
        
        # Check if prediction matches expected
        if result['prediction'] == article['expected']:
            correct_predictions += 1
            print("   ✅ Correct prediction!")
        else:
            print("   ❌ Incorrect prediction")
    
    print("\n" + "=" * 40)
    print(f"📊 Demo Results: {correct_predictions}/{total_articles} correct predictions")
    accuracy = (correct_predictions / total_articles) * 100
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    
    print("\n🚀 Features Demonstrated:")
    print("• Real-time text analysis")
    print("• Confidence scoring")
    print("• Probability distributions")
    print("• Text preprocessing")
    print("• Machine learning predictions")
    
    print("\n💡 Try the web application for more features:")
    print("• Interactive dashboard")
    print("• Data visualizations")
    print("• API endpoints")
    print("• Modern UI/UX")
    print("• Comprehensive analytics")

if __name__ == "__main__":
    main()
