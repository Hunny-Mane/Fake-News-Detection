#!/usr/bin/env python3
"""
Test script for Fake News Detection Application
"""

import requests
import json
import time

def test_home_page():
    """Test the home page is accessible"""
    try:
        response = requests.get('http://localhost:5000/')
        if response.status_code == 200:
            print("✅ Home page is accessible")
            return True
        else:
            print(f"❌ Home page returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the application. Make sure it's running on http://localhost:5000")
        return False

def test_dashboard():
    """Test the dashboard page is accessible"""
    try:
        response = requests.get('http://localhost:5000/dashboard')
        if response.status_code == 200:
            print("✅ Dashboard is accessible")
            return True
        else:
            print(f"❌ Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the dashboard")
        return False

def test_about_page():
    """Test the about page is accessible"""
    try:
        response = requests.get('http://localhost:5000/about')
        if response.status_code == 200:
            print("✅ About page is accessible")
            return True
        else:
            print(f"❌ About page returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the about page")
        return False

def test_api_predict():
    """Test the API prediction endpoint"""
    # Sample news articles for testing
    test_cases = [
        {
            "title": "Scientists Discover New Species of Deep Sea Creatures",
            "content": "A team of marine biologists has discovered several new species of deep sea creatures in the Pacific Ocean. The discovery was made during a research expedition using advanced underwater drones. The findings have been published in the prestigious Journal of Marine Biology."
        },
        {
            "title": "BREAKING: Aliens Contact Earth Government",
            "content": "Incredible news! Aliens have made contact with world leaders and are sharing advanced technology. This is absolutely real and not fake at all. The government is hiding this from us but we have inside sources."
        }
    ]
    
    try:
        for i, test_case in enumerate(test_cases, 1):
            response = requests.post(
                'http://localhost:5000/api/predict',
                json=test_case,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API Test {i} - Prediction: {result['prediction']}, Confidence: {result['confidence']:.2f}")
                print(f"   True Probability: {result['true_probability']:.2f}")
                print(f"   Fake Probability: {result['fake_probability']:.2f}")
            else:
                print(f"❌ API Test {i} failed with status code: {response.status_code}")
                return False
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON response from API")
        return False

def test_api_stats():
    """Test the API stats endpoint"""
    try:
        response = requests.get('http://localhost:5000/api/stats')
        if response.status_code == 200:
            stats = response.json()
            if 'error' not in stats:
                print("✅ API Stats endpoint working")
                print(f"   Total Articles: {stats['total_articles']}")
                print(f"   True Articles: {stats['true_articles']}")
                print(f"   Fake Articles: {stats['fake_articles']}")
                return True
            else:
                print(f"❌ API Stats error: {stats['error']}")
                return False
        else:
            print(f"❌ API Stats returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API Stats")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON response from API Stats")
        return False

def test_form_prediction():
    """Test the form-based prediction"""
    test_data = {
        'title': 'NASA Announces New Mars Mission',
        'content': 'NASA has announced plans for a new Mars mission scheduled for 2025. The mission will focus on searching for signs of ancient microbial life and collecting samples for return to Earth. This represents a significant step forward in our understanding of the Red Planet.'
    }
    
    try:
        response = requests.post('http://localhost:5000/predict-form', data=test_data)
        if response.status_code == 200:
            print("✅ Form prediction endpoint working")
            return True
        else:
            print(f"❌ Form prediction failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to form prediction")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Fake News Detection Application")
    print("=" * 50)
    
    tests = [
        ("Home Page", test_home_page),
        ("Dashboard", test_dashboard),
        ("About Page", test_about_page),
        ("API Prediction", test_api_predict),
        ("API Stats", test_api_stats),
        ("Form Prediction", test_form_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the application setup.")
    
    return passed == total

if __name__ == "__main__":
    main()
