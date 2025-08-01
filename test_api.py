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

# ==============================================================================
# UI Input API Tests - Các test mới cho UI input APIs
# ==============================================================================

def test_ui_input():
    """Test UI input processing endpoint"""
    print("\n🔍 Testing UI input processing...")
    
    # Test data
    input_data = {
        "input_text": "person walking in red shirt",
        "input_type": "search",
        "user_id": "user123",
        "session_id": "session456"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ui/input",
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ UI input processing completed")
            print(f"   Success: {data.get('success')}")
            print(f"   Processed text: {data.get('processed_text')}")
            print(f"   Suggestions: {len(data.get('suggestions', []))} items")
            return True
        else:
            print(f"❌ UI input processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ UI input processing error: {e}")
        return False

def test_text_processing():
    """Test text processing endpoint"""
    print("\n🔍 Testing text processing...")
    
    # Test data
    text_data = {
        "text": "Person walking in red shirt on sunny day",
        "processing_type": "extract",
        "language": "en"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ui/text/process",
            json=text_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Text processing completed")
            print(f"   Original: {data.get('original_text')}")
            print(f"   Processed: {data.get('processed_text')}")
            print(f"   Confidence: {data.get('confidence'):.2f}")
            return True
        else:
            print(f"❌ Text processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text processing error: {e}")
        return False

def test_query_suggestions():
    """Test query suggestions endpoint"""
    print("\n🔍 Testing query suggestions...")
    
    # Test data
    suggestion_data = {
        "partial_query": "person",
        "context": ["video", "walking"],
        "max_suggestions": 3
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ui/query/suggest",
            json=suggestion_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Query suggestions completed")
            print(f"   Partial query: {data.get('partial_query')}")
            print(f"   Suggestions: {data.get('suggestions')}")
            print(f"   Total suggestions: {data.get('total_suggestions')}")
            return True
        else:
            print(f"❌ Query suggestions failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Query suggestions error: {e}")
        return False

def test_filter_input():
    """Test filter input processing endpoint"""
    print("\n🔍 Testing filter input processing...")
    
    # Test data
    filter_data = {
        "filter_type": "object",
        "filter_values": ["person", "car", "building"],
        "operator": "AND",
        "priority": 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ui/filter/input",
            json=filter_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Filter input processing completed")
            print(f"   Filter type: {data.get('filter_type')}")
            print(f"   Processed filters: {data.get('processed_filters')}")
            print(f"   Validation status: {data.get('validation_status')}")
            return True
        else:
            print(f"❌ Filter input processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Filter input processing error: {e}")
        return False

def test_batch_input():
    """Test batch input processing endpoint"""
    print("\n🔍 Testing batch input processing...")
    
    # Test data
    batch_data = {
        "inputs": [
            "person walking",
            "red car driving",
            "building with windows"
        ],
        "batch_type": "search",
        "priority": "normal"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ui/batch/input",
            json=batch_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch input processing completed")
            print(f"   Total inputs: {data.get('total_inputs')}")
            print(f"   Processed: {data.get('processed_inputs')}")
            print(f"   Failed: {data.get('failed_inputs')}")
            print(f"   Processing time: {data.get('processing_time'):.3f}s")
            return True
        else:
            print(f"❌ Batch input processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch input processing error: {e}")
        return False

def test_input_types():
    """Test input types endpoint"""
    print("\n🔍 Testing input types...")
    
    try:
        response = requests.get(f"{BASE_URL}/ui/input/types")
        if response.status_code == 200:
            data = response.json()
            print("✅ Input types retrieved")
            print(f"   Available input types: {len(data.get('input_types', []))}")
            print(f"   Processing types: {data.get('processing_types')}")
            return True
        else:
            print(f"❌ Input types failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Input types error: {e}")
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
        test_compare_search,
        # UI Input API tests
        test_ui_input,
        test_text_processing,
        test_query_suggestions,
        test_filter_input,
        test_batch_input,
        test_input_types
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