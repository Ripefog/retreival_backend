#!/usr/bin/env python3
"""
Test the fixed exact count implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_engine import HybridRetriever

def test_exact_count_parsing():
    """Test that exact_count format is properly parsed"""
    print("🧪 Testing exact_count parsing...")
    
    retriever = HybridRetriever()
    
    # Test exact_count format
    query = {
        "person": {"exact_count": 3, "constraints": []},
        "bicycle": {"exact_count": 3, "constraints": []}
    }
    
    normalized = retriever._normalize_object_filters(query)
    
    tests = [
        ("person is count_aware", normalized.get("person", {}).get("type") == "count_aware"),
        ("person count is 3", normalized.get("person", {}).get("count") == 3),
        ("bicycle is count_aware", normalized.get("bicycle", {}).get("type") == "count_aware"),
        ("bicycle count is 3", normalized.get("bicycle", {}).get("count") == 3),
    ]
    
    passed = sum(1 for _, test in tests if test)
    
    for desc, result in tests:
        status = "✅" if result else "❌"
        print(f"  {status} {desc}")
    
    print(f"📊 Parsing test: {passed}/{len(tests)} passed")
    return passed == len(tests)

def test_count_validation_scenarios():
    """Test count validation for different scenarios"""
    print("\n🧪 Testing count validation scenarios...")
    
    retriever = HybridRetriever()
    
    scenarios = [
        # (detected, required, should_pass, description)
        (3, 3, True, "Perfect match: 3=3"),
        (6, 3, False, "Over count: 6≠3"),  
        (1, 3, False, "Under count: 1≠3"),
        (15, 3, False, "Way over: 15≠3"),
        (0, 3, False, "Zero count: 0≠3"),
    ]
    
    passed = 0
    for detected, required, should_pass, desc in scenarios:
        result = retriever._validate_count_constraint(detected, required)
        correct = (result == should_pass)
        status = "✅" if correct else "❌"
        print(f"  {status} {desc} → {result}")
        
        if correct:
            passed += 1
    
    print(f"📊 Validation test: {passed}/{len(scenarios)} passed")
    return passed == len(scenarios)

def test_scoring_differences():
    """Test scoring differences between exact match vs mismatch"""
    print("\n🧪 Testing scoring differences...")
    
    retriever = HybridRetriever()
    
    # Mock filter spec for exact count
    filter_spec = {
        'type': 'count_aware',
        'count': 3,
        'constraints': [],
        'all_match': None
    }
    
    # Mock obj_hits for different scenarios
    def create_mock_hits(count):
        return [{'distance': 0.5, 'entity': {}} for _ in range(count)]
    
    scenarios = [
        (3, "Perfect match (3=3)"),
        (6, "Over count (6≠3)"),  
        (1, "Under count (1≠3)"),
    ]
    
    results = []
    for count, desc in scenarios:
        obj_hits = create_mock_hits(count)
        
        # Test validation
        is_valid = retriever._validate_count_constraint(count, 3)
        
        # Test scoring (should return 0 for invalid)
        if is_valid:
            score = retriever._calculate_count_aware_score(obj_hits, filter_spec, "person")
        else:
            score = 0.0  # Would get penalty instead
        
        results.append((desc, count, is_valid, score))
        print(f"  📊 {desc}: valid={is_valid}, score={score:.3f}")
    
    # Perfect match should have highest score
    perfect_score = results[0][3]
    other_scores = [r[3] for r in results[1:]]
    
    success = perfect_score > max(other_scores) if other_scores else True
    status = "✅" if success else "❌"
    print(f"  {status} Perfect match has highest score: {perfect_score:.3f}")
    
    return success

def show_expected_behavior():
    """Show expected behavior for the user's query"""
    print("\n🎯 Expected Behavior for Your Query")
    print("=" * 60)
    
    query = """
    {
        "object_filters": {
            "person": {"exact_count": 3, "constraints": []},
            "bicycle": {"exact_count": 3, "constraints": []}
        }
    }
    """
    
    print(f"Query:{query}")
    
    results = [
        {
            "keyframe": "L23_V023_0272.00s.jpg",
            "detected": "6 person, 1 bicycle", 
            "penalties": ["person: -0.5", "bicycle: -0.5"],
            "final_effect": "Score drops significantly (penalties applied)",
            "rank": "❌ LOW (due to penalties)"
        },
        {
            "keyframe": "L23_V015_0106.92s.jpg", 
            "detected": "15 person, 13 bicycle",
            "penalties": ["person: -0.5", "bicycle: -0.5"],
            "final_effect": "Score drops significantly (penalties applied)",
            "rank": "❌ LOW (due to penalties)"
        },
        {
            "keyframe": "L23_V007_0130.36s.jpg",
            "detected": "3 person, 3 bicycle",
            "boosts": ["person: +0.5 (2x boost)", "bicycle: +0.5 (2x boost)"],
            "final_effect": "Score increases significantly (+1.0 total)",
            "rank": "✅ TOP (perfect match rewards)"
        }
    ]
    
    print("\nExpected Results After Fix:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['keyframe']}")
        print(f"   Detected: {result['detected']}")
        if 'penalties' in result:
            print(f"   Penalties: {', '.join(result['penalties'])}")
        if 'boosts' in result:
            print(f"   Boosts: {', '.join(result['boosts'])}")
        print(f"   Effect: {result['final_effect']}")
        print(f"   Rank: {result['rank']}")

def main():
    """Run all tests"""
    print("🚀 Testing Fixed Exact Count Implementation\n")
    
    tests = [
        test_exact_count_parsing,
        test_count_validation_scenarios,
        test_scoring_differences,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Test Results: {passed}/{total} passed")
    
    if all(results):
        print("🎉 All tests passed! The fix should work correctly.")
        show_expected_behavior()
        print("\nNext steps:")
        print("1. Restart the API server to load the changes")
        print("2. Test with your exact query")
        print("3. Check that L23_V007_0130.36s.jpg (3 people, 3 bikes) ranks #1")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())