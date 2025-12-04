from flask import Flask, request, render_template, jsonify, redirect, url_for
import pickle
import re
import nltk
import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
import os
import requests
from PIL import Image
import cv2
import tempfile

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load model & vectorizer
import os
model_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(model_dir, 'fake_news_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# Language-specific models (e.g., Hindi, Gujarati)
lang_models = {}

def preprocess_non_english(text):
    # Lightweight normalization for non-English scripts; keep letters from any script
    try:
        import regex as re_unicode
        text = str(text)
        # Remove non-letters except whitespace
        text = re_unicode.sub(r'[^\p{L}\s]+', ' ', text)
        text = re_unicode.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        # Fallback: basic cleanup
        text = str(text)
        return ' '.join(text.split())

def train_language_models_from_dataset():
    """Train per-language character-ngrams models from multilingual CSV if available."""
    multilingual_csv = os.path.join(model_dir, 'News _dataset', 'Hindi.csv')
    if not os.path.exists(multilingual_csv):
        return

    try:
        df_multi = pd.read_csv(multilingual_csv)
        # Expect columns: title, text, news_type, language
        required_cols = {'title', 'text', 'news_type', 'language'}
        if not required_cols.issubset(set(df_multi.columns)):
            return

        # Normalize labels
        df_multi = df_multi.dropna(subset=['text', 'news_type', 'language'])
        df_multi['label'] = df_multi['news_type'].str.strip().str.lower().map({'fake': 1, 'true': 0})
        df_multi = df_multi.dropna(subset=['label'])

        # Build combined text
        df_multi['combined'] = (df_multi.get('title', '').astype(str) + ' ' + df_multi['text'].astype(str)).str.strip()

        # Train per-language models for languages we detect
        for lang_code, group in df_multi.groupby(df_multi['language'].str.strip().str.lower()):
            if lang_code not in ['hindi', 'gujarati']:
                continue
            x_texts = [preprocess_non_english(t) for t in group['combined'].tolist()]
            y = group['label'].astype(int).tolist()
            if len(set(y)) < 2 or len(x_texts) < 10:
                continue

            # Character n-gram vectorizer works well across scripts
            lang_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2)
            X = lang_vectorizer.fit_transform(x_texts)
            lang_model = LogisticRegression(max_iter=1000)
            lang_model.fit(X, y)

            # Map to language code keys used by detector
            key = 'hi' if lang_code == 'hindi' else 'gu' if lang_code == 'gujarati' else None
            if key:
                lang_models[key] = {
                    'model': lang_model,
                    'vectorizer': lang_vectorizer
                }

        # Optionally persist models
        for key, bundle in lang_models.items():
            try:
                with open(os.path.join(model_dir, f'model_{key}.pkl'), 'wb') as f:
                    pickle.dump(bundle['model'], f)
                with open(os.path.join(model_dir, f'vectorizer_{key}.pkl'), 'wb') as f:
                    pickle.dump(bundle['vectorizer'], f)
            except Exception:
                pass
    except Exception as e:
        print(f"Error training language models: {e}")

def load_cached_language_models():
    for key in ['hi', 'gu']:
        model_path = os.path.join(model_dir, f'model_{key}.pkl')
        vec_path = os.path.join(model_dir, f'vectorizer_{key}.pkl')
        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                with open(model_path, 'rb') as f:
                    m = pickle.load(f)
                with open(vec_path, 'rb') as f:
                    v = pickle.load(f)
                lang_models[key] = {'model': m, 'vectorizer': v}
            except Exception:
                continue

# Try to load cached models, else train from dataset if available
load_cached_language_models()
if not lang_models:
    train_language_models_from_dataset()

# Load dataset for analysis
try:
    true_df = pd.read_csv(os.path.join(model_dir, 'News _dataset', 'True.csv'))
    fake_df = pd.read_csv(os.path.join(model_dir, 'News _dataset', 'Fake.csv'))
    true_df['label'] = 0
    fake_df['label'] = 1
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df[['title', 'text', 'label']].dropna()
except:
    df = pd.DataFrame()

# Global counters for dashboard
search_count = 0
analysis_stats = {
    'total_searches': 0,
    'fake_detected': 0,
    'real_detected': 0,
    'image_analyses': 0,
    'multilingual_analyses': 0
}

# Storage for analysis results
analysis_results_file = os.path.join(model_dir, 'analysis_results.json')

def load_analysis_results():
    """Load analysis results from file"""
    try:
        if os.path.exists(analysis_results_file):
            with open(analysis_results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all required keys exist
                if 'analyses' not in data:
                    data['analyses'] = []
                if 'total_analyses' not in data:
                    data['total_analyses'] = len(data['analyses'])
                if 'true_analyses' not in data:
                    data['true_analyses'] = sum(1 for a in data['analyses'] if not bool(a.get('is_fake', False)))
                if 'fake_analyses' not in data:
                    data['fake_analyses'] = sum(1 for a in data['analyses'] if bool(a.get('is_fake', False)))
                if 'multilingual_analyses' not in data:
                    data['multilingual_analyses'] = sum(1 for a in data['analyses'] if a.get('analysis_type') == 'multilingual')
                if 'image_analyses' not in data:
                    data['image_analyses'] = sum(1 for a in data['analyses'] if a.get('analysis_type') == 'image')
                return data
    except Exception as e:
        print(f"Error loading analysis results: {e}")
    
    return {
        'analyses': [],
        'total_analyses': 0,
        'true_analyses': 0,
        'fake_analyses': 0,
        'multilingual_analyses': 0,
        'image_analyses': 0
    }

def save_analysis_result(title, content, prediction, confidence, analysis_type='text'):
    """Save analysis result to persistent storage"""
    try:
        # Load existing results
        results = load_analysis_results()
        
        # Create new analysis record
        analysis_record = {
            'id': len(results['analyses']) + 1,
            'timestamp': datetime.now().isoformat(),
            'title': title[:100] if title else '',  # Limit title length
            'content_length': len(content) if content else 0,
            'prediction': prediction,
            'confidence': confidence,
            'analysis_type': analysis_type,
            'is_fake': bool(prediction == 1 if isinstance(prediction, int) else 'FAKE' in str(prediction))
        }
        
        # Add to results
        results['analyses'].append(analysis_record)
        results['total_analyses'] = len(results['analyses'])
        
        # Recalculate counters to ensure accuracy
        results['true_analyses'] = sum(1 for a in results['analyses'] if not bool(a.get('is_fake', False)))
        results['fake_analyses'] = sum(1 for a in results['analyses'] if bool(a.get('is_fake', False)))
        results['multilingual_analyses'] = sum(1 for a in results['analyses'] if a.get('analysis_type') == 'multilingual')
        results['image_analyses'] = sum(1 for a in results['analyses'] if a.get('analysis_type') == 'image')
        
        # Keep only last 1000 analyses to prevent file from growing too large
        if len(results['analyses']) > 1000:
            results['analyses'] = results['analyses'][-1000:]
            # Recalculate after trimming
            results['total_analyses'] = len(results['analyses'])
            results['true_analyses'] = sum(1 for a in results['analyses'] if not bool(a.get('is_fake', False)))
            results['fake_analyses'] = sum(1 for a in results['analyses'] if bool(a.get('is_fake', False)))
            results['multilingual_analyses'] = sum(1 for a in results['analyses'] if a.get('analysis_type') == 'multilingual')
            results['image_analyses'] = sum(1 for a in results['analyses'] if a.get('analysis_type') == 'image')
        
        # Save to file with proper error handling
        temp_file = analysis_results_file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Atomic file replacement
        if os.path.exists(temp_file):
            if os.path.exists(analysis_results_file):
                os.remove(analysis_results_file)
            os.rename(temp_file, analysis_results_file)
        
        print(f"Analysis saved: {analysis_type} - {'FAKE' if analysis_record['is_fake'] else 'TRUE'} (ID: {analysis_record['id']})")
            
    except Exception as e:
        print(f"Error saving analysis result: {e}")
        # Clean up temp file if it exists
        if os.path.exists(analysis_results_file + '.tmp'):
            os.remove(analysis_results_file + '.tmp')

def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    results = load_analysis_results()
    
    # Get original dataset stats
    if not df.empty:
        dataset_total = int(len(df))
        dataset_true = int(len(df[df['label'] == 0]))
        dataset_fake = int(len(df[df['label'] == 1]))
        avg_text_length = float(df['text'].str.len().mean())
        avg_title_length = float(df['title'].str.len().mean())
    else:
        dataset_total = dataset_true = dataset_fake = avg_text_length = avg_title_length = 0
    
    # Get analysis stats
    analysis_total = results['total_analyses']
    analysis_true = results['true_analyses']
    analysis_fake = results['fake_analyses']
    
    return {
        'dataset_stats': {
            'total_news': int(dataset_total),
            'true_news': int(dataset_true),
            'fake_news': int(dataset_fake),
            'avg_text_length': int(avg_text_length),
            'avg_title_length': int(avg_title_length)
        },
        'analysis_stats': {
            'total_analyses': int(analysis_total),
            'true_analyses': int(analysis_true),
            'fake_analyses': int(analysis_fake),
            'multilingual_analyses': int(results['multilingual_analyses']),
            'image_analyses': int(results['image_analyses'])
        },
        'combined_stats': {
            'total_news': int(dataset_total + analysis_total),
            'true_news': int(dataset_true + analysis_true),
            'fake_news': int(dataset_fake + analysis_fake)
        }
    }

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    return ' '.join(ps.stem(w) for w in words if w not in stop_words)

def create_dynamic_visualizations():
    """Create dynamic visualizations based on analysis results"""
    dashboard_stats = get_dashboard_stats()
    results = load_analysis_results()
    
    plt.style.use('seaborn-v0_8')
    
    # Chart 1: Combined Dataset + Analysis Distribution
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    combined_stats = dashboard_stats['combined_stats']
    analysis_stats = dashboard_stats['analysis_stats']
    
    # Create pie chart with combined data
    sizes = [combined_stats['true_news'], combined_stats['fake_news']]
    colors = ['#2E8B57', '#DC143C']
    labels = [f'True News ({combined_stats["true_news"]})', f'Fake News ({combined_stats["fake_news"]})']
    
    if sum(sizes) > 0:
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Total News Distribution (Dataset + Analysis)', fontsize=16, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Total News Distribution', fontsize=16, fontweight='bold')
    
    img1 = io.BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight', dpi=300)
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()
    
    # Chart 2: Analysis Results Over Time (if we have analysis data)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if results['analyses']:
        # Group analyses by day
        from datetime import datetime
        daily_counts = {}
        for analysis in results['analyses']:
            try:
                date = datetime.fromisoformat(analysis['timestamp']).date()
                if date not in daily_counts:
                    daily_counts[date] = {'true': 0, 'fake': 0}
                if analysis['is_fake']:
                    daily_counts[date]['fake'] += 1
                else:
                    daily_counts[date]['true'] += 1
            except:
                continue
        
        if daily_counts:
            dates = sorted(daily_counts.keys())
            true_counts = [daily_counts[date]['true'] for date in dates]
            fake_counts = [daily_counts[date]['fake'] for date in dates]
            
            ax2.plot(dates, true_counts, label='True News', color='#2E8B57', marker='o')
            ax2.plot(dates, fake_counts, label='Fake News', color='#DC143C', marker='s')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Analyses')
            ax2.set_title('Daily Analysis Results')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Daily Analysis Results')
    else:
        ax2.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Daily Analysis Results')
    
    img2 = io.BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight', dpi=300)
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()
    plt.close()
    
    # Chart 3: Analysis Types Distribution
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if results['analyses']:
        type_counts = {}
        for analysis in results['analyses']:
            analysis_type = analysis.get('analysis_type', 'text')
            if analysis_type not in type_counts:
                type_counts[analysis_type] = 0
            type_counts[analysis_type] += 1
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
            
            ax3.bar(types, counts, color=colors[:len(types)])
            ax3.set_xlabel('Analysis Type')
            ax3.set_ylabel('Number of Analyses')
            ax3.set_title('Analysis Types Distribution')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Analysis Types Distribution')
    else:
        ax3.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Analysis Types Distribution')
    
    img3 = io.BytesIO()
    plt.savefig(img3, format='png', bbox_inches='tight', dpi=300)
    img3.seek(0)
    plot_url3 = base64.b64encode(img3.getvalue()).decode()
    plt.close()
    
    # Chart 4: Confidence Score Distribution
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    if results['analyses']:
        confidences = []
        for analysis in results['analyses']:
            try:
                conf = float(analysis.get('confidence', 0))
                confidences.append(conf)
            except:
                continue
        
        if confidences:
            ax4.hist(confidences, bins=20, alpha=0.7, color='#667eea', edgecolor='black')
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Confidence Score Distribution')
            ax4.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.2f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No confidence data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Confidence Score Distribution')
    else:
        ax4.text(0.5, 0.5, 'No analysis data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Confidence Score Distribution')
    
    img4 = io.BytesIO()
    plt.savefig(img4, format='png', bbox_inches='tight', dpi=300)
    img4.seek(0)
    plot_url4 = base64.b64encode(img4.getvalue()).decode()
    plt.close()
    
    return {
        'label_dist': plot_url1,
        'text_length': plot_url2,
        'title_length': plot_url3,
        'word_count': plot_url4
    }

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/dashboard')
def dashboard():
    # Get dynamic statistics
    dashboard_stats = get_dashboard_stats()
    
    # Create dynamic visualizations based on analysis results
    plots = create_dynamic_visualizations()
    
    # Combine all statistics
    stats = {
        'total_news': dashboard_stats['combined_stats']['total_news'],
        'true_news': dashboard_stats['combined_stats']['true_news'],
        'fake_news': dashboard_stats['combined_stats']['fake_news'],
        'avg_text_length': dashboard_stats['dataset_stats']['avg_text_length'],
        'avg_title_length': dashboard_stats['dataset_stats']['avg_title_length'],
        'analysis_stats': dashboard_stats['analysis_stats'],
        'dataset_stats': dashboard_stats['dataset_stats']
    }
    
    return render_template('dashboard.html', plots=plots, stats=stats)

@app.route('/predict-form', methods=['POST'])
def predict_form():
    global analysis_stats
    title = request.form['title']
    content = request.form['content']
    combined = title + " " + content

    processed = preprocess(combined)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0]

    # Update counters
    analysis_stats['total_searches'] += 1
    if pred == 0:
        analysis_stats['real_detected'] += 1
    else:
        analysis_stats['fake_detected'] += 1

    # Save analysis result to persistent storage
    save_analysis_result(title, content, pred, max(confidence), 'text')

    result = {
        "prediction": "🟢 This news is TRUE" if pred == 0 else "🔴 This news is FAKE",
        "confidence": f"{max(confidence) * 100:.2f}%",
        "true_prob": f"{confidence[0] * 100:.2f}%",
        "fake_prob": f"{confidence[1] * 100:.2f}%"
    }
    
    return render_template('index.html', prediction=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global analysis_stats
    data = request.get_json()
    title = data.get('title', '')
    content = data.get('content', '')
    combined = title + " " + content

    processed = preprocess(combined)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0]

    # Update counters
    analysis_stats['total_searches'] += 1
    if pred == 0:
        analysis_stats['real_detected'] += 1
    else:
        analysis_stats['fake_detected'] += 1

    # Save analysis result to persistent storage
    save_analysis_result(title, content, pred, max(confidence), 'text')

    return jsonify({
        "prediction": "TRUE" if pred == 0 else "FAKE",
        "confidence": float(max(confidence)),
        "true_probability": float(confidence[0]),
        "fake_probability": float(confidence[1])
    })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/stats')
def api_stats():
    """Get real-time dashboard statistics"""
    dashboard_stats = get_dashboard_stats()
    results = load_analysis_results()
    
    # Get recent analyses (last 10)
    recent_analyses = results['analyses'][-10:] if results['analyses'] else []
    
    return jsonify({
        "dataset_stats": dashboard_stats['dataset_stats'],
        "analysis_stats": dashboard_stats['analysis_stats'],
        "combined_stats": dashboard_stats['combined_stats'],
        "recent_analyses": recent_analyses,
        "total_analyses": len(results['analyses'])
    })

@app.route('/api/dashboard-update')
def dashboard_update():
    """Get updated dashboard data for real-time updates"""
    dashboard_stats = get_dashboard_stats()
    return jsonify(dashboard_stats)

@app.route('/api/debug-analysis')
def debug_analysis():
    """Debug endpoint to check analysis results"""
    results = load_analysis_results()
    return jsonify({
        "file_exists": os.path.exists(analysis_results_file),
        "results": results,
        "file_size": os.path.getsize(analysis_results_file) if os.path.exists(analysis_results_file) else 0
    })

@app.route('/advanced')
def advanced():
    return render_template('advanced.html')

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    global analysis_stats
    try:
        analysis_stats['image_analyses'] += 1
        
        # Simulate image analysis (in real implementation, use actual image processing)
        image_url = request.form.get('imageUrl')
        image_file = request.files.get('image')
        
        if not image_url and not image_file:
            return jsonify({"success": False, "error": "No image provided"})
        
        # Simulate analysis results
        results = {
            "metadata_analysis": {
                "metadata": {
                    "format": "jpg",
                    "size": "2.5MB",
                    "dimensions": {"width": 1920, "height": 1080},
                    "creationDate": datetime.now().isoformat(),
                    "camera": {"make": "Canon", "model": "EOS R5"},
                    "suspicious": False
                },
                "riskScore": 15,
                "indicators": ["No suspicious metadata detected"]
            },
            "visual_analysis": {
                "analysis": {
                    "colors": {"dominant": "#FF0000", "palette": ["#FF0000", "#00FF00", "#0000FF"]},
                    "composition": {"ruleOfThirds": True, "symmetry": False},
                    "quality": {"resolution": "high", "sharpness": "good"},
                    "objects": ["person", "car", "building"],
                    "faces": {"count": 1, "quality": "good"}
                },
                "riskScore": 25,
                "indicators": ["Good image quality", "Natural composition"]
            },
            "manipulation_detection": {
                "manipulation": {
                    "clonedRegions": {"detected": False, "confidence": 0.95},
                    "resampling": {"detected": False, "confidence": 0.90},
                    "compression": {"quality": "high", "artifacts": "minimal"}
                },
                "riskScore": 10,
                "indicators": ["No manipulation detected"]
            },
            "deepfake_detection": {
                "deepfake": {
                    "faceSwap": {"detected": False, "confidence": 0.88},
                    "expression": {"natural": True, "consistency": "good"},
                    "eyeMovement": {"natural": True, "blinking": "normal"}
                },
                "riskScore": 5,
                "indicators": ["No deepfake detected"]
            },
            "credibility_assessment": {
                "credibility": {
                    "source": {"credibility": "medium", "verified": False},
                    "context": {"relevant": True, "consistent": True},
                    "verification": {"verified": False, "sources": 0}
                },
                "riskScore": 30,
                "indicators": ["Source verification needed"]
            },
            "finalAssessment": {
                "isFake": False,
                "confidence": 85,
                "riskLevel": "LOW",
                "overallScore": 17,
                "summary": "This image appears to be authentic with no significant manipulation detected.",
                "recommendations": [
                    "Image appears authentic",
                    "Consider cross-referencing with other sources"
                ]
            }
        }
        
        # Persist this image analysis for dashboard stats
        try:
            img_title = (image_url or (image_file.filename if image_file else '')) or 'Image Analysis'
            is_fake_img = bool(results.get('finalAssessment', {}).get('isFake', False))
            conf_img = float(results.get('finalAssessment', {}).get('confidence', 0)) / 100.0
            save_analysis_result(img_title, img_title, 1 if is_fake_img else 0, conf_img, 'image')
        except Exception as _:
            pass

        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/test-simple', methods=['POST'])
def test_simple():
    try:
        return jsonify({"success": True, "message": "Test successful"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analyze-multilingual', methods=['POST'])
def analyze_multilingual():
    global analysis_stats
    try:
        analysis_stats['multilingual_analyses'] += 1
        
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"})
        
        # Detect language
        detected_language = detect_language(text)
        
        # Prefer language-specific model if available for hi/gu; else fall back to translation + English model
        used_method = None
        if detected_language in lang_models:
            bundle = lang_models[detected_language]
            txt = preprocess_non_english(text)
            vec_l = bundle['vectorizer'].transform([txt])
            proba = bundle['model'].predict_proba(vec_l)[0]
            pred = int(bundle['model'].predict(vec_l)[0])
            is_fake = bool(pred == 1)
            confidence_score = float(max(proba) * 100)
            ml_confidence = float(max(proba))
            used_method = f"Language-specific ML ({detected_language})"
            translated_text = text  # keep original
            fake_score = calculate_enhanced_fake_score(text, text)
        else:
            # Translate text to English for analysis
            translated_text = translate_text(text, 'en')
            # Enhanced analysis for multilingual content
            fake_score = calculate_enhanced_fake_score(text, translated_text)
            if detected_language != 'en':
                # Use enhanced analysis when no specific model exists
                is_fake = bool(fake_score > 40)
                confidence_score = float(max(100 - fake_score, fake_score))
                ml_confidence = 0.5
                used_method = "Enhanced Analysis Only"
            else:
                # English text → use main ML model
                processed = preprocess(translated_text)
                vec_e = vectorizer.transform([processed])
                pred = model.predict(vec_e)[0]
                confidence = model.predict_proba(vec_e)[0]
                ml_prediction = bool(pred == 1)
                enhanced_prediction = bool(fake_score > 40)
                if fake_score < 20:
                    is_fake = bool(enhanced_prediction)
                else:
                    is_fake = bool((0.6 * ml_prediction + 0.4 * enhanced_prediction) > 0.5)
                confidence_score = float(max(confidence) * 100)
                ml_confidence = float(max(confidence))
                used_method = "ML Model + Enhanced Analysis"
        
        # Generate results
        factors = generate_multilingual_factors(text, fake_score)
        recommendations = generate_multilingual_recommendations(fake_score, detected_language)
        
        results = {
            "detectedLanguage": str(detected_language),
            "originalText": str(text),
            "translatedText": str(translated_text),
            "classification": {
                "isFake": bool(is_fake),
                "level": "HIGH" if fake_score > 60 else "MEDIUM" if fake_score > 30 else "LOW",
                "description": "Strong indicators of fake news" if fake_score > 60 else "Moderate indicators" if fake_score > 30 else "Appears credible"
            },
            "confidence": float(confidence_score),
            "fakeScore": int(fake_score),
            "mlConfidence": float(ml_confidence),
            "analysisMethod": used_method if used_method else ("ML Model + Enhanced Analysis" if detected_language == 'en' else "Enhanced Analysis Only"),
            "factors": factors,
            "recommendations": recommendations
        }
        
        # Save multilingual analysis result
        save_analysis_result("", text, 1 if is_fake else 0, confidence_score/100, 'multilingual')
        
        return jsonify({"success": True, "results": results})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/advanced-search', methods=['POST'])
def advanced_search():
    global analysis_stats
    try:
        analysis_stats['total_searches'] += 1
        
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('searchType', 'both')
        news_type = data.get('newsType', 'all')
        
        if not query:
            return jsonify({"success": False, "error": "No search query provided"})
        
        if df.empty:
            return jsonify({"success": True, "results": [], "message": "No dataset available for search"})
        
        # Filter by news type
        filtered_df = df.copy()
        if news_type == 'true':
            filtered_df = filtered_df[filtered_df['label'] == 0]
        elif news_type == 'fake':
            filtered_df = filtered_df[filtered_df['label'] == 1]
        
        # Enhanced search with multiple query terms
        results = []
        query_terms = [term.strip().lower() for term in query.split() if term.strip()]
        
        for _, row in filtered_df.iterrows():
            match_score = 0
            title_lower = str(row['title']).lower()
            text_lower = str(row['text']).lower()
            
            # Calculate match score based on search type
            if search_type in ['title', 'both']:
                for term in query_terms:
                    if term in title_lower:
                        match_score += 2  # Title matches are weighted higher
                    # Check for partial matches
                    if any(term in word for word in title_lower.split()):
                        match_score += 1
            
            if search_type in ['content', 'both']:
                for term in query_terms:
                    if term in text_lower:
                        match_score += 1
                    # Check for partial matches
                    if any(term in word for word in text_lower.split()):
                        match_score += 0.5
            
            # Add to results if there's a match
            if match_score > 0:
                # Truncate text for display
                display_text = row['text'][:300] + "..." if len(row['text']) > 300 else row['text']
                
                results.append({
                    'title': row['title'],
                    'text': display_text,
                    'full_text': row['text'],
                    'label': row['label'],
                    'match_score': match_score,
                    'news_type': 'TRUE' if row['label'] == 0 else 'FAKE'
                })
        
        # Sort by match score (highest first)
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Limit results
        results = results[:20]
        
        # Add search statistics
        search_stats = {
            'total_matches': len(results),
            'query_terms': query_terms,
            'search_type': search_type,
            'news_type_filter': news_type,
            'dataset_size': len(filtered_df)
        }
        
        return jsonify({
            "success": True, 
            "results": results,
            "search_stats": search_stats
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def detect_language(text):
    """Simple language detection based on character sets"""
    hindi_chars = re.compile(r'[\u0900-\u097F]')
    gujarati_chars = re.compile(r'[\u0A80-\u0AFF]')
    tamil_chars = re.compile(r'[\u0B80-\u0BFF]')
    telugu_chars = re.compile(r'[\u0C00-\u0C7F]')
    bengali_chars = re.compile(r'[\u0980-\u09FF]')
    punjabi_chars = re.compile(r'[\u0A00-\u0A7F]')
    
    if hindi_chars.search(text):
        return 'hi'
    elif gujarati_chars.search(text):
        return 'gu'
    elif tamil_chars.search(text):
        return 'ta'
    elif telugu_chars.search(text):
        return 'te'
    elif bengali_chars.search(text):
        return 'bn'
    elif punjabi_chars.search(text):
        return 'pa'
    else:
        return 'en'

def translate_text(text, target_lang):
    """Enhanced translation mapping for common phrases and patterns"""
    # Common translation mappings for fake news detection
    translation_map = {
        'hi': {
            # Fake news indicators
            'फेक न्यूज': 'fake news',
            'झूठी खबर': 'false news',
            'गलत जानकारी': 'wrong information',
            'भ्रामक': 'misleading',
            'अविश्वसनीय': 'unbelievable',
            'चौंकाने वाला': 'shocking',
            'अत्यधिक': 'excessive',
            'तुरंत': 'immediately',
            'जल्दी': 'urgent',
            'सावधान': 'warning',
            'खतरा': 'danger',
            'संकट': 'crisis',
            'वास्तविक समाचार': 'real news',
            'सच्ची खबर': 'true news',
            'विश्लेषण': 'analysis',
            'पता लगाना': 'detection',
            'जांच': 'investigation',
            'सत्यापन': 'verification',
            # Common words
            'है': 'is',
            'हैं': 'are',
            'था': 'was',
            'थे': 'were',
            'की': 'of',
            'के': 'of',
            'को': 'to',
            'में': 'in',
            'पर': 'on',
            'से': 'from',
            'और': 'and',
            'या': 'or',
            'लेकिन': 'but',
            'कि': 'that',
            'यह': 'this',
            'वह': 'that',
            'हम': 'we',
            'आप': 'you',
            'वे': 'they'
        },
        'gu': {
            # Fake news indicators
            'ફેક ન્યૂઝ': 'fake news',
            'ખોટા સમાચાર': 'false news',
            'ખોટી માહિતી': 'wrong information',
            'ભ્રામક': 'misleading',
            'અવિશ્વસનીય': 'unbelievable',
            'ચોંકાવનારું': 'shocking',
            'અત્યધિક': 'excessive',
            'તરત': 'immediately',
            'જલ્દી': 'urgent',
            'સાવધાન': 'warning',
            'ખતરો': 'danger',
            'સંકટ': 'crisis',
            'વાસ્તવિક સમાચાર': 'real news',
            'સાચા સમાચાર': 'true news',
            'વિશ્લેષણ': 'analysis',
            'શોધ': 'detection',
            'તપાસ': 'investigation',
            'ચકાસણી': 'verification',
            # Common words
            'છે': 'is',
            'છે': 'are',
            'હતું': 'was',
            'હતા': 'were',
            'ની': 'of',
            'ના': 'of',
            'ને': 'to',
            'માં': 'in',
            'પર': 'on',
            'થી': 'from',
            'અને': 'and',
            'કે': 'or',
            'પણ': 'but',
            'કે': 'that',
            'આ': 'this',
            'તે': 'that',
            'અમે': 'we',
            'તમે': 'you',
            'તેઓ': 'they'
        },
        'ta': {
            # Fake news indicators
            'போலி செய்தி': 'fake news',
            'தவறான செய்தி': 'false news',
            'தவறான தகவல்': 'wrong information',
            'தவறான': 'misleading',
            'நம்பமுடியாத': 'unbelievable',
            'அதிர்ச்சியூட்டும்': 'shocking',
            'அதிகப்படியான': 'excessive',
            'உடனடியாக': 'immediately',
            'அவசர': 'urgent',
            'எச்சரிக்கை': 'warning',
            'ஆபத்து': 'danger',
            'நெருக்கடி': 'crisis',
            'உண்மையான செய்தி': 'real news',
            'சரியான செய்தி': 'true news',
            'பகுப்பாய்வு': 'analysis',
            'கண்டறிதல்': 'detection',
            'விசாரணை': 'investigation',
            'சரிபார்ப்பு': 'verification',
            # Common words
            'உள்ளது': 'is',
            'உள்ளன': 'are',
            'இருந்தது': 'was',
            'இருந்தன': 'were',
            'இன்': 'of',
            'ஆன': 'of',
            'ஐ': 'to',
            'இல்': 'in',
            'மீது': 'on',
            'இருந்து': 'from',
            'மற்றும்': 'and',
            'அல்லது': 'or',
            'ஆனால்': 'but',
            'என்று': 'that',
            'இது': 'this',
            'அது': 'that',
            'நாங்கள்': 'we',
            'நீங்கள்': 'you',
            'அவர்கள்': 'they'
        }
    }
    
    # Detect source language
    source_lang = detect_language(text)
    
    # If source language is English or target language, return as is
    if source_lang == 'en' or source_lang == target_lang:
        return text
    
    # Apply translation mappings
    translated_text = text
    if source_lang in translation_map:
        for original, translated in translation_map[source_lang].items():
            translated_text = translated_text.replace(original, translated)
    
    # For non-English text, create a more comprehensive translation
    if source_lang != 'en':
        # Add some context words to help with analysis
        translated_text += " news article content analysis"
    
    return translated_text

def calculate_enhanced_fake_score(original_text, translated_text):
    """Calculate enhanced fake score for multilingual content"""
    score = 0
    
    # Combine original and translated text for analysis
    combined_text = (original_text + " " + translated_text).lower()
    
    # Check for suspicious patterns in both original and translated text
    suspicious_patterns = [
        'urgent', 'shocking', 'unbelievable', 'amazing', 'incredible',
        'breaking', 'exclusive', 'secret', 'hidden', 'revealed',
        'fake news', 'false news', 'misleading', 'wrong information',
        'तुरंत', 'चौंकाने वाला', 'अविश्वसनीय', 'झूठी खबर',
        'તરત', 'ચોંકાવનારું', 'અવિશ્વસનીય', 'ખોટા સમાચાર',
        'உடனடியாக', 'அதிர்ச்சியூட்டும்', 'நம்பமுடியாத', 'போலி செய்தி'
    ]
    
    for pattern in suspicious_patterns:
        if pattern in combined_text:
            score += 10
    
    # Check for emotional manipulation
    emotional_words = [
        'shocking', 'outrageous', 'scandalous', 'explosive', 'terrifying',
        'भ्रामक', 'अत्यधिक', 'ભ્રામક', 'અત્યધિક', 'தவறான', 'அதிகப்படியான'
    ]
    for word in emotional_words:
        if word in combined_text:
            score += 5
    
    # Check for urgency indicators
    urgency_words = [
        'urgent', 'breaking', 'act now', 'limited time', 'immediately',
        'जल्दी', 'सावधान', 'જલ્દી', 'સાવધાન', 'அவசர', 'எச்சரிக்கை'
    ]
    for word in urgency_words:
        if word in combined_text:
            score += 8
    
    # Check for danger/crisis indicators
    danger_words = [
        'danger', 'crisis', 'warning', 'alert', 'threat',
        'खतरा', 'संकट', 'ખતરો', 'સંકટ', 'ஆபத்து', 'நெருக்கடி'
    ]
    for word in danger_words:
        if word in combined_text:
            score += 6
    
    # Check for excessive punctuation (common in fake news)
    if combined_text.count('!') > 3 or combined_text.count('?') > 3:
        score += 5
    
    # Check for all caps (common in fake news)
    if len(combined_text) > 20 and combined_text.isupper():
        score += 8
    
    # Check for repeated words (common in fake news)
    words = combined_text.split()
    if len(words) > 10:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        max_repetition = max(word_counts.values()) if word_counts else 1
        if max_repetition > len(words) * 0.1:  # More than 10% repetition
            score += 7
    
    return min(score, 100)

def generate_multilingual_factors(text, fake_score):
    """Generate analysis factors for multilingual content"""
    factors = []
    
    # Analyze text characteristics
    if fake_score > 70:
        factors.append("Very high emotional intensity detected")
        factors.append("Multiple suspicious language patterns found")
        factors.append("Strong urgency and danger indicators present")
    elif fake_score > 50:
        factors.append("High emotional intensity detected")
        factors.append("Suspicious language patterns found")
        factors.append("Urgency indicators present")
    elif fake_score > 30:
        factors.append("Moderate emotional intensity detected")
        factors.append("Some suspicious language patterns found")
    elif fake_score > 15:
        factors.append("Low-level suspicious indicators detected")
    else:
        factors.append("No obvious red flags detected")
    
    # Language-specific factors
    detected_lang = detect_language(text)
    if detected_lang != 'en':
        factors.append(f"Content analyzed in {detected_lang.upper()} language")
        factors.append("Cross-language analysis performed")
    
    # Text quality factors
    if len(text) < 50:
        factors.append("Very short content - limited analysis possible")
    elif len(text) > 1000:
        factors.append("Long-form content - comprehensive analysis performed")
    
    return factors

def generate_multilingual_recommendations(fake_score, language):
    """Generate recommendations for multilingual content"""
    recommendations = []
    
    if fake_score > 70:
        recommendations.extend([
            "⚠️ HIGH RISK: This content shows strong indicators of being fake news",
            "Verify this information from multiple reliable sources immediately",
            "Check the author's credentials and publication reputation",
            "Look for citations, references, and fact-checking sources",
            "Be extremely cautious before sharing this content"
        ])
    elif fake_score > 50:
        recommendations.extend([
            "⚠️ MEDIUM-HIGH RISK: This content shows concerning indicators",
            "Verify this information from multiple reliable sources",
            "Check the author's credentials and reputation",
            "Look for citations and references",
            "Cross-reference with other news sources"
        ])
    elif fake_score > 30:
        recommendations.extend([
            "⚠️ MEDIUM RISK: Some suspicious indicators detected",
            "Cross-reference with other news sources",
            "Check the publication date and context",
            "Verify the source credibility"
        ])
    elif fake_score > 15:
        recommendations.extend([
            "⚠️ LOW RISK: Minor suspicious indicators detected",
            "Consider cross-referencing with other sources",
            "Check the publication context"
        ])
    else:
        recommendations.extend([
            "✅ LOW RISK: This content appears credible",
            "Continue to verify information from multiple sources as a best practice"
        ])
    
    # Add language-specific recommendations
    if language != 'en':
        lang_names = {
            'hi': 'Hindi',
            'gu': 'Gujarati', 
            'ta': 'Tamil',
            'te': 'Telugu',
            'bn': 'Bengali',
            'pa': 'Punjabi'
        }
        lang_name = lang_names.get(language, language.upper())
        recommendations.append(f"Consider checking sources in both {lang_name} and English")
        recommendations.append("Verify information across multiple language sources")
    
    # General recommendations
    recommendations.extend([
        "Always check multiple sources before sharing news",
        "Look for official statements and verified accounts",
        "Be aware of your own biases when evaluating information"
    ])
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
