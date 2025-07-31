#!/usr/bin/env python3
"""
Simple test script for Video Retrieval Backend API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Milvus: {data.get('milvus', {}).get('status')}")
            print(f"   Elasticsearch: {data.get('elasticsearch', {}).get('status')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_search_modes():
    """Test search modes endpoint"""
    print("\n🔍 Testing search modes...")
    try:
        response = requests.get(f"{BASE_URL}/search/modes")
        if response.status_code == 200:
            data = response.json()
            print("✅ Search modes retrieved")
            print(f"   Available modes: {data.get('modes')}")
            return True
        else:
            print(f"❌ Search modes failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search modes error: {e}")
        return False

def test_collections():
    """Test collections endpoint"""
    print("\n🔍 Testing collections...")
    try:
        response = requests.get(f"{BASE_URL}/collections")
        if response.status_code == 200:
            data = response.json()
            print("✅ Collections info retrieved")
            collections = data.get('collections', {})
            for name, info in collections.items():
                print(f"   {name}: {info.get('num_entities', 0)} entities")
            return True
        else:
            print(f"❌ Collections failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Collections error: {e}")
        return False

def test_search():
    """Test search endpoint"""
    print("\n🔍 Testing search...")
    
    # Test data
    search_data = {
        "text_query": "person walking",
        "mode": "hybrid",
        "object_filters": ["person"],
        "color_filters": ["red"],
        "top_k": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/search",
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Search completed")
            print(f"   Query: {data.get('query')}")
            print(f"   Mode: {data.get('mode')}")
            print(f"   Results: {data.get('total_results')}")
            
            # Show first result
            results = data.get('results', [])
            if results:
                first_result = results[0]
                print(f"   First result: {first_result.get('keyframe_id')} (score: {first_result.get('score'):.3f})")
            
            return True
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_compare_search():
    """Test compare search endpoint"""
    print("\n🔍 Testing compare search...")
    
    # Test data
    search_data = {
        "text_query": "person walking",
        "top_k": 3
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/search/compare",
            json=search_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Compare search completed")
            print(f"   Query: {data.get('query')}")
            
            comparison = data.get('comparison', {})
            for mode, result in comparison.items():
                print(f"   {mode}: {result.get('total_results')} results")
            
            return True
        else:
            print(f"❌ Compare search failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Compare search error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Video Retrieval Backend API Tests")
    print("=" * 50)
    
    # Wait for server to start
    print("⏳ Waiting for server to be ready...")
    time.sleep(5)
    
    # Run tests
    tests = [
        test_health,
        test_search_modes,
        test_collections,
        test_search,
        test_compare_search
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 