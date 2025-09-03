#!/usr/bin/env python3
"""
Test script to demonstrate the new exact count filtering feature.
"""

import requests
import json
from typing import Dict, Any

# API endpoint
API_BASE = "http://localhost:8000"

def test_exact_count_filtering():
    """Test the new exact count filtering feature"""
    
    test_cases = [
        {
            "name": "Exact count: 3 people",
            "payload": {
                "text_query": "group of people in meeting room",
                "mode": "hybrid",
                "object_filters": {
                    "person": {
                        "exact_count": 3,
                        "constraints": []
                    }
                },
                "top_k": 5
            }
        },
        {
            "name": "Count range: 2-4 chairs",
            "payload": {
                "text_query": "conference room with chairs",
                "mode": "hybrid", 
                "object_filters": {
                    "chair": {
                        "min_count": 2,
                        "max_count": 4,
                        "constraints": []
                    }
                },
                "top_k": 5
            }
        },
        {
            "name": "Exact count with constraints",
            "payload": {
                "text_query": "two people in red shirts",
                "mode": "hybrid",
                "object_filters": {
                    "person": {
                        "exact_count": 2,
                        "constraints": [
                            {"color": [255, 0, 0]},
                            {"color": [255, 0, 0], "bbox": [100, 100, 500, 400]}
                        ]
                    }
                },
                "top_k": 5
            }
        },
        {
            "name": "Legacy format (should still work)",
            "payload": {
                "text_query": "person sitting at desk",
                "mode": "hybrid",
                "object_filters": {
                    "person": []
                },
                "top_k": 5
            }
        },
        {
            "name": "Mixed count constraints",
            "payload": {
                "text_query": "office environment",
                "mode": "hybrid",
                "object_filters": {
                    "person": {
                        "exact_count": 1,
                        "constraints": []
                    },
                    "chair": {
                        "min_count": 1,
                        "max_count": 3,
                        "constraints": []
                    },
                    "laptop": []  # No count constraint, just presence
                },
                "top_k": 10
            }
        }
    ]
    
    print("🧪 Testing Exact Count Filtering Feature\n")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)
        print(f"Query: '{test_case['payload']['text_query']}'")
        print(f"Object filters: {json.dumps(test_case['payload']['object_filters'], indent=2)}")
        
        try:
            response = requests.post(
                f"{API_BASE}/search",
                json=test_case['payload'],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: Found {result['total_results']} results")
                
                # Show top result details
                if result['results']:
                    top_result = result['results'][0]
                    print(f"   Top result: {top_result['keyframe_id']}")
                    print(f"   Score: {top_result['score']}")
                    print(f"   Reasons: {top_result['reasons']}")
                else:
                    print("   No results found")
                    
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def test_optimization_stats():
    """Test the optimization stats endpoint"""
    print("\n\n📊 Testing Optimization Stats")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/optimization/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Optimization stats retrieved successfully:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")

def show_feature_overview():
    """Show overview of the new exact count filtering feature"""
    print("🎯 Exact Count Filtering Feature Overview")
    print("=" * 60)
    print("""
NEW FEATURES ADDED:

1. Exact Count Filtering:
   - Filter images with exactly N objects
   - Example: {"person": {"exact_count": 3, "constraints": []}}

2. Count Range Filtering:
   - Filter images with objects in a range
   - Example: {"chair": {"min_count": 2, "max_count": 4, "constraints": []}}

3. Combined Count + Constraints:
   - Exact count with color/bbox constraints
   - Example: {"person": {"exact_count": 2, "constraints": [{"color": [255,0,0]}]}}

4. Smart Scoring:
   - Bonus (+0.3) for exact count matches
   - Penalty (-0.4) for count mismatches
   - Reduced weight for mismatched objects (30% of normal)

5. Backward Compatibility:
   - All existing formats still work
   - Empty list [] now properly supported for name-only filtering

6. Performance Optimizations:
   - Similarity threshold (0.7) for valid object matching
   - Vectorized operations for count validation
   - Intelligent weight adjustment based on count accuracy

SUPPORTED FORMATS:
- Legacy: {"person": []}
- Color only: {"person": [[255,0,0]]}
- Bbox only: {"person": [[100,100,500,500]]}
- Full: {"person": [[[255,0,0], [100,100,500,500]]]}
- Dict: {"person": [{"color": [255,0,0], "bbox": [100,100,500,500]}]}
- Exact count: {"person": {"exact_count": 3, "constraints": []}}
- Count range: {"person": {"min_count": 2, "max_count": 4, "constraints": []}}
""")

if __name__ == "__main__":
    show_feature_overview()
    
    # Check if API is available
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is available, running tests...\n")
            test_exact_count_filtering()
            test_optimization_stats()
        else:
            print(f"❌ API health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please ensure the API server is running at http://localhost:8000")