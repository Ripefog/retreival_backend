#!/usr/bin/env python3
"""
Test script to debug constraint evaluation with your exact query
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_constraint_query():
    """Test your exact query with constraints"""
    
    query = {
        "text_query": "a shot with a high angle view of the riders. There are three riders in the frame pedaling in a straight line. All three riders are from the same team, wearing white jerseys and yellow and blue shorts. The first rider is wearing a white cap, the second is wearing a red cap, and the last rider is wearing a black cap",     
        "mode": "hybrid",
        "object_filters": {
            "person": {
                "exact_count": 3,
                "constraints": [
                    {"color": [60.53456351191002, -3.5961640483800905, -1.2472249757613518]},
                    {"color": [65.62002071226156, -2.842780961784863, -0.9926034801127415]},
                    {"color": [62.97830704556726, 0.5358639811413046, 0.7756950357389547]}
                ]
            },
            "bicycle": {
                "exact_count": 3,
                "constraints": [
                    {"color": [43.07171373675201, -1.849118382510062, -5.108752119680071]},
                    {"color": [38.668402621002954, -1.3598528796085196, -4.3772764125807555]},
                    {"color": [34.91296302729005, -1.0050034449321232, -6.314376951890233]}
                ]
            }
        },
        "top_k": 10  # Smaller for testing
    }
    
    print("🧪 Testing constraint evaluation with your exact query")
    print("=" * 60)
    print(f"Query object filters:")
    print(json.dumps(query["object_filters"], indent=2))
    
    try:
        print("\n📡 Sending request...")
        response = requests.post(
            f"{API_BASE}/search",
            json=query,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful: {result['total_results']} results")
            
            # Check for constraint-related reasons in top results
            print("\n🔍 Checking top results for constraint scoring:")
            
            constraint_found = False
            for i, item in enumerate(result['results'][:3]):
                print(f"\nResult {i+1}: {item['keyframe_id']}")
                print(f"  Score: {item['score']}")
                print(f"  Reasons: {item['reasons']}")
                
                # Check if constraints are mentioned in reasons
                for reason in item['reasons']:
                    if 'constraint' in reason.lower() or 'color' in reason.lower():
                        constraint_found = True
                        print(f"  🎯 FOUND CONSTRAINT SCORING: {reason}")
            
            if not constraint_found:
                print("\n❌ NO CONSTRAINT SCORING FOUND IN RESULTS")
                print("This suggests constraints are not being evaluated properly.")
                
                print("\n🔧 DEBUG CHECKLIST:")
                print("1. Check server logs for these messages:")
                print("   - 'Count-aware processing: person detected=X, required=3'")
                print("   - 'Evaluating individual constraints: [...]'") 
                print("   - 'Color constraint evaluation:'")
                print("   - 'Final scoring breakdown:'")
                print("\n2. If no logs appear:")
                print("   - Server may not have latest code")
                print("   - Logging not enabled")
                print("   - Query format not recognized")
                
            else:
                print(f"\n✅ CONSTRAINT SCORING WORKING!")
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request exception: {e}")
        print("Make sure the API server is running at http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_simple_constraint():
    """Test with simpler constraint to isolate issues"""
    
    print("\n" + "=" * 60)
    print("🧪 Testing simpler constraint for comparison")
    
    simple_query = {
        "text_query": "people riding bicycles",     
        "mode": "hybrid",
        "object_filters": {
            "person": {
                "exact_count": 3,
                "constraints": [
                    {"color": [255, 255, 255]},  # Simple RGB white
                    {},  # Any
                    {}   # Any
                ]
            }
        },
        "top_k": 5
    }
    
    print("Simple query:")
    print(json.dumps(simple_query["object_filters"], indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE}/search",
            json=simple_query,
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Simple query successful: {result['total_results']} results")
            
            for i, item in enumerate(result['results'][:2]):
                print(f"Result {i+1}: {item['keyframe_id']}")
                print(f"  Reasons: {item['reasons']}")
                
        else:
            print(f"❌ Simple query failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Simple query error: {e}")

def main():
    """Run debugging tests"""
    print("🔧 Constraint Debugging Tool")
    print("This will test your exact query to debug constraint evaluation\n")
    
    # Check if API is available
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is available\n")
        else:
            print(f"⚠️ API health check returned: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please start the API server first:")
        print("  cd D:\\HCM_AI_CHALLENGE\\retreival_backend")
        print("  python -m app.main")
        return
    
    # Run tests
    test_constraint_query()
    test_simple_constraint()
    
    print("\n📋 Next Steps:")
    print("1. Check the API server console logs for detailed debugging info")
    print("2. Look for log messages starting with '🔍' and 'Count-aware processing'")
    print("3. If no constraint-related logs appear, there may be an issue with:")
    print("   - Code not being loaded (restart server)")
    print("   - Query format not recognized")
    print("   - Logging not enabled")

if __name__ == "__main__":
    main()