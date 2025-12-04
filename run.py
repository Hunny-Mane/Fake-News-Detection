#!/usr/bin/env python3
"""
Startup script for Fake News Detection Application
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'sklearn', 'nltk', 
        'matplotlib', 'seaborn', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_files():
    """Check if model files exist"""
    required_files = ['fake_news_model.pkl', 'tfidf_vectorizer.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("Please run: python train_model.py")
        return False
    
    print("✅ Model files found")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def start_application():
    """Start the Flask application"""
    print("\n🚀 Starting Fake News Detection Application...")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        print("✅ Application imported successfully")
        print("🌐 Server will be available at: http://localhost:5000")
        print("📊 Dashboard: http://localhost:5000/dashboard")
        print("ℹ️  About: http://localhost:5000/about")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Failed to import application: {e}")
        return False
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return False

def main():
    """Main startup function"""
    print("🛡️ Fake News Detection System")
    print("=" * 30)
    
    # Check prerequisites
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("NLTK Data", download_nltk_data)
    ]
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        if not check_func():
            print(f"\n❌ {check_name} check failed. Please fix the issues above.")
            return False
    
    print("\n✅ All checks passed!")
    
    # Start the application
    return start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
