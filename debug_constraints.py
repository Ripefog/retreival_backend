#!/usr/bin/env python3
"""
Debug script to test constraint evaluation with detailed logging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_engine import HybridRetriever
import logging

def setup_logging():
    """Setup detailed logging to see constraint evaluation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_constraint_parsing():
    """Test constraint parsing"""
    print("🧪 Testing constraint parsing with colors...")
    
    retriever = HybridRetriever()
    
    # Test query with color constraints
    query = {
        "person": {
            "exact_count": 3,
            "constraints": [
                {"color": [255, 255, 255]},  # White
                {"color": [255, 0, 0]},      # Red  
                {}                            # Any
            ]
        },
        "bicycle": {
            "exact_count": 3,
            "constraints": []
        }
    }
    
    normalized = retriever._normalize_object_filters(query)
    
    print("\nNormalized object filters:")
    for obj_name, filter_spec in normalized.items():
        print(f"  {obj_name}:")
        print(f"    Type: {filter_spec.get('type')}")
        print(f"    Count: {filter_spec.get('count')}")
        print(f"    Constraints: {filter_spec.get('constraints')}")
        print(f"    All match: {filter_spec.get('all_match')}")

def test_color_conversion():
    """Test RGB to LAB conversion"""
    print("\n🎨 Testing color conversion...")
    
    retriever = HybridRetriever()
    
    test_colors = [
        [255, 255, 255],  # White
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [0, 0, 0],        # Black
    ]
    
    for rgb in test_colors:
        lab = retriever._rgb_to_lab(tuple(rgb))
        print(f"  RGB {rgb} → LAB {lab}")

def test_constraint_evaluation_mock():
    """Test constraint evaluation with mock data"""
    print("\n🔍 Testing constraint evaluation with mock data...")
    
    retriever = HybridRetriever()
    
    # Mock entity data (simulating Milvus object)
    mock_entity = {
        'color_lab': '50.0,20.0,30.0',  # Mock LAB color data
        'bbox_xyxy': '100,100,200,200'   # Mock bbox data
    }
    
    # Test constraints
    constraints = [
        {"color": [255, 0, 0]},  # Red constraint
        {"color": [255, 255, 255]},  # White constraint
        {}  # No constraint
    ]
    
    print("Mock entity data:")
    print(f"  color_lab: {mock_entity['color_lab']}")
    print(f"  bbox_xyxy: {mock_entity['bbox_xyxy']}")
    
    for i, constraint in enumerate(constraints):
        print(f"\nConstraint {i+1}: {constraint}")
        score = retriever._evaluate_single_constraint(mock_entity, constraint)
        print(f"  Constraint score: {score:.3f}")

def show_debugging_instructions():
    """Show instructions for debugging with real API"""
    print("\n📋 Debugging Instructions")
    print("=" * 50)
    print("""
To debug constraint evaluation with real data:

1. SETUP LOGGING:
   Add this to your app/main.py:
   
   import logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(message)s'
   )

2. TEST QUERY:
   Run your query and watch for these logs:
   
   - "Count-aware processing: person detected=X, required=Y"
   - "Evaluating individual constraints: [...]"
   - "Color constraint evaluation:"
   - "Entity keys: [...]"
   - "Final scoring breakdown:"

3. CHECK FOR ISSUES:
   
   a) If no "Count-aware processing" logs:
      → Query format not recognized as count-aware
   
   b) If "No individual constraints, using score 1.0":
      → Constraints not properly parsed
   
   c) If "No entity color data":  
      → Milvus objects missing color_lab field
   
   d) If "Invalid entity color length":
      → color_lab format incorrect
      
4. EXPECTED LOGS FOR WORKING CONSTRAINTS:
   
   Count-aware processing: person detected=3, required=3
   Evaluating individual constraints: [{'color': [255, 255, 255]}, {'color': [255, 0, 0]}, {}]
   Color constraint evaluation:
     Required color: [255, 255, 255]
     Entity keys: ['id', 'color_lab', 'bbox_xyxy', ...]
     Entity color_lab: '95.0,0.0,0.0'
     Parsed entity color: [95.0, 0.0, 0.0]
     Entity LAB: (95.0, 0.0, 0.0)
     Required LAB: (95.0, 0.0, 0.0)
     Color distance: 0.00, score: 1.000
   Final scoring breakdown:
     Constraint match: 0.850 (weight: 0.2)

""")

def main():
    """Run constraint debugging"""
    print("🔧 Constraint Evaluation Debugging Tool\n")
    
    setup_logging()
    
    test_constraint_parsing()
    test_color_conversion() 
    test_constraint_evaluation_mock()
    show_debugging_instructions()
    
    print("\n✅ Debugging setup complete!")
    print("Now restart your API server and run the query with color constraints.")
    print("Check the server logs for detailed constraint evaluation information.")

if __name__ == "__main__":
    main()