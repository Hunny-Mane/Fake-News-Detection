#!/usr/bin/env python3
"""
Test script for advanced features
"""

import requests
import json

def test_advanced_features():
    base_url = "http://localhost:5000"
    
    print("Testing Advanced Features...")
    
    # Test 1: Image Analysis
    print("\n1. Testing Image Analysis...")
    try:
        response = requests.post(f"{base_url}/api/analyze-image", 
                               data={"imageUrl": "https://example.com/test.jpg"})
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Image analysis working")
                print(f"   - Authenticity: {'FAKE' if data['results']['finalAssessment']['isFake'] else 'AUTHENTIC'}")
                print(f"   - Confidence: {data['results']['finalAssessment']['confidence']}%")
            else:
                print("❌ Image analysis failed:", data.get("error"))
        else:
            print("❌ Image analysis request failed:", response.status_code)
    except Exception as e:
        print("❌ Image analysis error:", str(e))
    
    # Test 2: Multilingual Analysis
    print("\n2. Testing Multilingual Analysis...")
    try:
        test_data = {
            "text": "This is a test news article in English",
            "language": "en"
        }
        response = requests.post(f"{base_url}/api/analyze-multilingual", 
                               json=test_data)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Multilingual analysis working")
                print(f"   - Detected Language: {data['results']['detectedLanguage']}")
                print(f"   - Classification: {'FAKE' if data['results']['classification']['isFake'] else 'REAL'}")
                print(f"   - Confidence: {data['results']['confidence']}%")
            else:
                print("❌ Multilingual analysis failed:", data.get("error"))
        else:
            print("❌ Multilingual analysis request failed:", response.status_code)
    except Exception as e:
        print("❌ Multilingual analysis error:", str(e))
    
    # Test 3: Advanced Search
    print("\n3. Testing Advanced Search...")
    try:
        search_data = {
            "query": "politics",
            "searchType": "title",
            "newsType": "all"
        }
        response = requests.post(f"{base_url}/api/advanced-search", 
                               json=search_data)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Advanced search working")
                print(f"   - Results found: {len(data['results'])}")
                if data['results']:
                    print(f"   - First result: {data['results'][0]['title'][:50]}...")
            else:
                print("❌ Advanced search failed:", data.get("error"))
        else:
            print("❌ Advanced search request failed:", response.status_code)
    except Exception as e:
        print("❌ Advanced search error:", str(e))
    
    # Test 4: Dashboard Stats
    print("\n4. Testing Dashboard Stats...")
    try:
        response = requests.get(f"{base_url}/api/stats")
        if response.status_code == 200:
            data = response.json()
            if "analysis_stats" in data:
                print("✅ Dashboard stats working")
                stats = data["analysis_stats"]
                print(f"   - Total searches: {stats['total_searches']}")
                print(f"   - Real detected: {stats['real_detected']}")
                print(f"   - Fake detected: {stats['fake_detected']}")
                print(f"   - Image analyses: {stats['image_analyses']}")
                print(f"   - Multilingual analyses: {stats['multilingual_analyses']}")
            else:
                print("❌ Dashboard stats not available")
        else:
            print("❌ Dashboard stats request failed:", response.status_code)
    except Exception as e:
        print("❌ Dashboard stats error:", str(e))
    
    print("\n✅ Advanced features testing completed!")

if __name__ == "__main__":
    test_advanced_features()
