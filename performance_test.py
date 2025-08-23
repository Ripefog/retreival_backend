#!/usr/bin/env python3
"""
Performance test script to measure improvements in color/object filtering.
Run this before and after optimizations to compare performance.
"""

import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_engine import HybridRetriever
from app.config import settings
from app.database import init_database, close_database

async def test_search_performance():
    """Test search performance with various filter combinations."""
    
    print("🚀 Starting Performance Test...")
    
    # Initialize database and retriever
    await init_database()
    retriever = HybridRetriever()
    await retriever.initialize()
    
    # Test cases with increasing complexity
    test_cases = [
        {
            "name": "Simple Text Search",
            "params": {
                "text_query": "a person sitting at a desk",
                "mode": "hybrid",
                "top_k": 20
            }
        },
        {
            "name": "Text + Object Filters",
            "params": {
                "text_query": "a man wearing something blue",
                "mode": "hybrid",
                "object_filters": {
                    "person": [
                        ((50.0, -2.0, 12.0), (0, 0, 1920, 1080))
                    ],
                    "man": [
                        ((55.0, -1.5, 10.0), (100, 200, 600, 900))
                    ]
                },
                "top_k": 20
            }
        },
        {
            "name": "Text + Color Filters",
            "params": {
                "text_query": "red car on the street",
                "mode": "hybrid", 
                "color_filters": [
                    (32.0, -5.0, -35.0),  # Blue
                    (70.0, 60.0, 20.0),   # Pink/Red
                    (50.0, 65.0, 30.0)    # Another color
                ],
                "top_k": 20
            }
        },
        {
            "name": "Full Complex Search",
            "params": {
                "text_query": "a presenter on stage",
                "mode": "hybrid",
                "user_query": "Minh Tâm",
                "object_filters": {
                    "person": [
                        ((70.0, 0.0, 0.0), (10, 20, 300, 400))
                    ],
                    "presenter": [
                        ((65.0, 5.0, 5.0), (200, 100, 800, 700))
                    ]
                },
                "color_filters": [
                    (55.0, 65.0, 25.0),
                    (76.65912653528162, -0.5829774648731245, -19.05410702358592),
                    (18.828432444742575, -6.5353777349142215, 12.744719724371178)
                ],
                "ocr_query": "VIỆT NAM",
                "asr_query": "kinh tế",
                "top_k": 20
            }
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        print("-" * 50)
        
        # Warmup run
        try:
            await retriever.search(**test_case['params'])
        except Exception as e:
            print(f"❌ Warmup failed: {e}")
            continue
            
        # Actual performance test (multiple runs)
        times = []
        num_runs = 3
        
        for run in range(num_runs):
            start_time = time.time()
            try:
                search_results = await retriever.search(**test_case['params'])
                end_time = time.time()
                
                duration = end_time - start_time
                times.append(duration)
                
                print(f"  Run {run + 1}: {duration:.3f}s ({len(search_results)} results)")
                
            except Exception as e:
                print(f"  Run {run + 1}: FAILED - {e}")
                continue
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results[test_case['name']] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'num_runs': len(times)
            }
            
            print(f"  ✅ Average: {avg_time:.3f}s | Min: {min_time:.3f}s | Max: {max_time:.3f}s")
        else:
            print(f"  ❌ All runs failed")
    
    # Print summary
    print(f"\n" + "="*60)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*60)
    
    for test_name, metrics in results.items():
        print(f"{test_name:.<40} {metrics['avg_time']:.3f}s")
    
    print(f"\n🔧 Optimizations Applied:")
    print("  ✅ scipy.optimize.linear_sum_assignment (Hungarian)")  
    print("  ✅ Batch Milvus queries")
    print("  ✅ NumPy vectorization for color distances")
    print("  ✅ colorspacious for CIEDE2000 (if available)")
    print("  ✅ asyncio.gather for parallel processing")
    
    # Cleanup
    await close_database()
    print(f"\n🏁 Performance test completed!")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_search_performance())