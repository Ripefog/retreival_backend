#!/usr/bin/env python3
"""
Unit test script for exact count matching functionality in object filters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_engine import HybridRetriever
import json

def test_count_constraint_validation():
    """Test count constraint validation logic"""
    print("🧪 Testing count constraint validation...")
    
    retriever = HybridRetriever()
    
    # Test cases
    test_cases = [
        # (detected_count, constraint, expected_result)
        (3, 3, True),           # Exact match
        (2, 3, False),          # Under count
        (4, 3, False),          # Over count
        (3, [2, 4], True),      # In range
        (1, [2, 4], False),     # Below range
        (5, [2, 4], False),     # Above range
        (3, ">=3", True),       # Greater equal
        (2, ">=3", False),      # Less than
        (3, "<=3", True),       # Less equal
        (4, "<=3", False),      # Greater than
        (3, "!=2", True),       # Not equal
        (2, "!=2", False),      # Equal (should fail)
        (5, None, True),        # No constraint
    ]
    
    passed = 0
    failed = 0
    
    for detected, constraint, expected in test_cases:
        result = retriever._validate_count_constraint(detected, constraint)
        status = "✅" if result == expected else "❌"
        print(f"  {status} detected={detected}, constraint={constraint}, expected={expected}, got={result}")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"📊 Count validation tests: {passed} passed, {failed} failed")
    return failed == 0

def test_count_accuracy_scoring():
    """Test count accuracy scoring logic"""
    print("\n🧪 Testing count accuracy scoring...")
    
    retriever = HybridRetriever()
    
    test_cases = [
        # (detected, required, expected_score_range)
        (3, 3, (0.95, 1.0)),       # Perfect match
        (2, 3, (0.4, 0.6)),        # Close but not exact
        (1, 3, (0.0, 0.2)),        # Far from target
        (3, [2, 4], (0.95, 1.0)),  # In range, near center
        (2, [2, 4], (0.8, 0.9)),   # In range, edge
        (4, [2, 4], (0.8, 0.9)),   # In range, edge
        (1, [2, 4], (0.0, 0.5)),   # Out of range
        (3, ">=3", (0.95, 1.0)),   # Expression match
        (2, ">=3", (0.0, 0.1)),    # Expression fail
        (5, None, (0.95, 1.0)),    # No constraint
    ]
    
    passed = 0
    failed = 0
    
    for detected, required, (min_score, max_score) in test_cases:
        score = retriever._calculate_count_accuracy_score(detected, required)
        in_range = min_score <= score <= max_score
        status = "✅" if in_range else "❌"
        print(f"  {status} detected={detected}, required={required}, score={score:.3f}, expected=[{min_score}, {max_score}]")
        
        if in_range:
            passed += 1
        else:
            failed += 1
    
    print(f"📊 Count accuracy tests: {passed} passed, {failed} failed")
    return failed == 0

def test_object_filter_parsing():
    """Test object filter parsing and normalization"""
    print("\n🧪 Testing object filter parsing...")
    
    retriever = HybridRetriever()
    
    # Test count-aware format
    count_aware_filter = {
        "person": {
            "count": 3,
            "constraints": [
                {"color": [255, 0, 0]},
                {"bbox": [100, 100, 200, 200]},
                {}
            ]
        }
    }
    
    normalized = retriever._normalize_object_filters(count_aware_filter)
    person_spec = normalized.get("person", {})
    
    checks = [
        ("Filter type is count_aware", person_spec.get("type") == "count_aware"),
        ("Count constraint is 3", person_spec.get("count") == 3),
        ("Has 3 constraints", len(person_spec.get("constraints", [])) == 3),
        ("First constraint has color", "color" in person_spec.get("constraints", [{}])[0]),
        ("Second constraint has bbox", "bbox" in person_spec.get("constraints", [{}])[1]),
        ("Third constraint is empty", person_spec.get("constraints", [{}])[2] == {}),
    ]
    
    passed = sum(1 for _, check in checks if check)
    failed = len(checks) - passed
    
    for desc, check in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {desc}")
    
    # Test legacy format
    legacy_filter = {
        "person": [
            [[255, 0, 0], [100, 100, 200, 200]]
        ]
    }
    
    normalized_legacy = retriever._normalize_object_filters(legacy_filter)
    legacy_spec = normalized_legacy.get("person", {})
    
    legacy_checks = [
        ("Legacy filter type is legacy", legacy_spec.get("type") == "legacy"),
        ("Has constraints", len(legacy_spec.get("constraints", [])) > 0),
    ]
    
    for desc, check in legacy_checks:
        status = "✅" if check else "❌"
        print(f"  {status} {desc}")
        if check:
            passed += 1
        else:
            failed += 1
    
    print(f"📊 Filter parsing tests: {passed} passed, {failed} failed")
    return failed == 0

def test_example_queries():
    """Test example query formats from models.py"""
    print("\n🧪 Testing example query formats...")
    
    retriever = HybridRetriever()
    
    # Test exact count example
    exact_count_query = {
        "person": {
            "count": 3,
            "constraints": [
                {"color": [255, 0, 0]},
                {"bbox": [100, 100, 300, 400]},
                {}
            ]
        }
    }
    
    # Test range count example  
    range_count_query = {
        "person": {
            "count": [3, 6],
            "all_match": {"bbox": [0, 0, 800, 400]}
        }
    }
    
    queries = [
        ("Exact count query", exact_count_query),
        ("Range count query", range_count_query),
    ]
    
    passed = 0
    failed = 0
    
    for desc, query in queries:
        try:
            normalized = retriever._normalize_object_filters(query)
            person_spec = normalized.get("person", {})
            is_count_aware = person_spec.get("type") == "count_aware"
            has_count = person_spec.get("count") is not None
            
            if is_count_aware and has_count:
                print(f"  ✅ {desc} - parsed successfully")
                print(f"      Count: {person_spec.get('count')}")
                print(f"      Constraints: {len(person_spec.get('constraints', []))}")
                print(f"      All match: {person_spec.get('all_match') is not None}")
                passed += 1
            else:
                print(f"  ❌ {desc} - parsing failed")
                print(f"      Type: {person_spec.get('type')}")
                print(f"      Count: {person_spec.get('count')}")
                failed += 1
        except Exception as e:
            print(f"  ❌ {desc} - exception: {e}")
            failed += 1
    
    print(f"📊 Example query tests: {passed} passed, {failed} failed")
    return failed == 0

def show_implementation_summary():
    """Show implementation summary"""
    print("🎯 Exact Count Matching Implementation Summary")
    print("=" * 60)
    print("""
✅ COMPLETED FEATURES:

1. Count Constraint Types:
   - Exact count: count: 3
   - Range count: count: [2, 4] 
   - Expression count: count: ">=2"

2. Object Filter Formats:
   - count: Constraint for exact/range/expression count
   - constraints: Individual constraint specifications
   - all_match: Constraint that all objects must satisfy

3. Validation Logic:
   - _validate_count_constraint(): Validates detected count vs requirement
   - _calculate_count_accuracy_score(): Scores accuracy of count matching

4. Scoring System:
   - Count accuracy weight: 40%
   - Semantic match weight: 30%
   - Constraint match weight: 20%
   - Count bonus weight: 10%

5. Parser & Normalization:
   - _normalize_count_aware_filter(): Handles new format
   - _normalize_legacy_filter(): Backward compatibility
   - Unified processing in _normalize_object_filters()

6. Integration:
   - Updated main object filtering pipeline
   - Route to count-aware vs legacy processing
   - Comprehensive scoring and reasons

EXAMPLE USAGE:

# Exact count: Find images with exactly 3 people
{
    "text_query": "group photo",
    "object_filters": {
        "person": {
            "count": 3,
            "constraints": [
                {"color": [255, 0, 0]},    # One red person
                {"bbox": [100, 100, 300, 400]}, # One in specific area
                {}                          # One anywhere
            ]
        }
    }
}

# Range count: Find images with 2-4 chairs
{
    "text_query": "conference room", 
    "object_filters": {
        "chair": {
            "count": [2, 4],
            "all_match": {"bbox": [0, 0, 800, 400]}  # All in upper half
        }
    }
}

BACKWARD COMPATIBILITY:
- All existing formats still work unchanged
- Legacy processing path preserved
- Seamless migration for users
""")

def main():
    """Run all tests"""
    print("🚀 Starting exact count matching tests...\n")
    
    show_implementation_summary()
    
    tests = [
        test_count_constraint_validation,
        test_count_accuracy_scoring, 
        test_object_filter_parsing,
        test_example_queries,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"\n🎯 Overall results: {passed_tests}/{total_tests} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Exact count matching is ready for use.")
        print("\nNext steps:")
        print("1. Start the API server: python -m app.main")
        print("2. Test with real queries using the new count format")
        print("3. Monitor performance and accuracy metrics")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())