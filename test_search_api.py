#!/usr/bin/env python3
"""
Test script for /search API endpoint of Video Retrieval Backend.
Mocks database responses (Milvus & Elasticsearch) and missing BEiT3 checkpoint
while executing real AI models (MetaCLIP 2) on GPU.
"""

import sys
import os
import time
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure app is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fastapi.testclient import TestClient

def create_mock_milvus_hits():
    """Generates realistic mock Milvus search results for testing retrieval engine."""
    return [
        {
            "id": "L01_V001_12.50s.jpg",
            "distance": 0.85,
            "entity": {
                "keyframe_id": "L01_V001_12.50s.jpg",
                "timestamp": 12.50,
                "object_ids": "101, 102, 103",
                "lab_colors": "70.0,15.0,25.0, 50.0,-10.0,30.0, 80.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0",
            }
        },
        {
            "id": "L01_V001_45.20s.jpg",
            "distance": 0.78,
            "entity": {
                "keyframe_id": "L01_V001_45.20s.jpg",
                "timestamp": 45.20,
                "object_ids": "104, 105",
                "lab_colors": "60.0,20.0,40.0, 40.0,5.0,10.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0",
            }
        },
        {
            "id": "L01_V002_100.00s.jpg",
            "distance": 0.71,
            "entity": {
                "keyframe_id": "L01_V002_100.00s.jpg",
                "timestamp": 100.00,
                "object_ids": "106",
                "lab_colors": "55.0,65.0,25.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0",
            }
        }
    ]

def create_mock_object_hits():
    """Generates mock object Milvus hits for count/color/bbox filter tests."""
    return {
        101: {"id": 101, "distance": 0.15, "entity": {"bbox_xyxy": "100, 100, 300, 400", "color_lab": "70.0, 15.0, 25.0"}},
        102: {"id": 102, "distance": 0.20, "entity": {"bbox_xyxy": "350, 100, 550, 400", "color_lab": "50.0, -10.0, 30.0"}},
        103: {"id": 103, "distance": 0.18, "entity": {"bbox_xyxy": "600, 100, 800, 400", "color_lab": "80.0, 0.0, 0.0"}},
        104: {"id": 104, "distance": 0.30, "entity": {"bbox_xyxy": "50, 50, 200, 200", "color_lab": "60.0, 20.0, 40.0"}},
        105: {"id": 105, "distance": 0.35, "entity": {"bbox_xyxy": "250, 50, 400, 200", "color_lab": "40.0, 5.0, 10.0"}},
        106: {"id": 106, "distance": 0.40, "entity": {"bbox_xyxy": "10, 10, 100, 100", "color_lab": "55.0, 65.0, 25.0"}}
    }

def run_tests():
    print("=" * 70)
    print("🚀 API ENDPOINT TEST: POST /search")
    print("=" * 70)

    from app.config import settings
    from app.database import db_manager
    from app.metaclip2 import MetaCLIP2Encoder

    # Patch connect_all and collection loaders before importing app
    db_manager.connect_all = AsyncMock(return_value=None)
    db_manager.connect_milvus = AsyncMock(return_value=None)
    db_manager.connect_elasticsearch = AsyncMock(return_value=None)
    db_manager._load_milvus_collections = AsyncMock(return_value=None)
    db_manager.milvus_connected = True
    db_manager.elasticsearch_connected = True
    db_manager.get_collection = MagicMock(return_value=MagicMock())
    db_manager.check_milvus_connection = MagicMock(return_value={"status": "connected"})
    db_manager.check_elasticsearch_connection = MagicMock(return_value={"status": "connected"})

    from app.retrieval_engine import HybridRetriever

    # Custom _load_models that loads MetaCLIP2 on GPU without failing on missing BEiT3 file
    def mock_load_models(self):
        print("  Loading MetaCLIP 2 onto GPU...")
        self.metaclip2 = MetaCLIP2Encoder(
            model_name=settings.METACLIP2_MODEL_NAME,
            device=self.device,
            expected_dim=settings.METACLIP2_DIM,
            cache_dir=settings.HF_CACHE_DIR,
        )
        print("  ✅ MetaCLIP 2 loaded successfully on GPU!")

    HybridRetriever._load_models = mock_load_models

    from app.main import app

    print(f"Device: {settings.DEVICE.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mock_hits = create_mock_milvus_hits()
    mock_obj_dict = create_mock_object_hits()

    with patch.object(HybridRetriever, '_search_milvus_async', new_callable=AsyncMock) as mock_search_milvus, \
         patch.object(HybridRetriever, '_batch_search_milvus_objects', new_callable=AsyncMock) as mock_batch_objs, \
         patch.object(HybridRetriever, '_async_hybrid_reranking', new_callable=AsyncMock) as mock_rerank:

        mock_search_milvus.return_value = mock_hits
        mock_batch_objs.return_value = mock_obj_dict
        mock_rerank.return_value = None

        with TestClient(app) as client:
            print("\n--- Test 1: GET /health ---")
            res = client.get("/health")
            print(f"Status Code: {res.status_code}")
            print(f"Response: {res.json()}")
            assert res.status_code == 200, "Health check failed!"
            print("✅ GET /health PASSED!")

            print("\n--- Test 2: POST /search (Simple Hybrid Text Query) ---")
            payload1 = {
                "text_query": "a person walking in a red shirt",
                "mode": "hybrid",
                "top_k": 5
            }
            t0 = time.time()
            res1 = client.post("/search", json=payload1)
            dt1 = (time.time() - t0) * 1000
            print(f"Status Code: {res1.status_code} ({dt1:.2f}ms)")
            body1 = res1.json()
            print(f"Returned Query: \"{body1.get('query')}\"")
            print(f"Total Results: {body1.get('total_results')}")
            for r in body1.get("results", []):
                print(f"  - [{r['keyframe_id']}] Score: {r['score']}, Reasons: {r['reasons']}")
            assert res1.status_code == 200, "Search query failed!"
            assert body1.get("total_results") > 0, "No search results returned!"
            print("✅ Test 2 PASSED!")

            print("\n--- Test 3: POST /search (Count-Aware Object Filters) ---")
            payload2 = {
                "text_query": "group photo of three people",
                "mode": "metaclip2",
                "object_filters": {
                    "person": {
                        "count": 3,
                        "constraints": [
                            {"color": [255, 0, 0]},
                            {"bbox": [100, 100, 300, 400]},
                            {}
                        ]
                    }
                },
                "top_k": 5
            }
            t0 = time.time()
            res2 = client.post("/search", json=payload2)
            dt2 = (time.time() - t0) * 1000
            print(f"Status Code: {res2.status_code} ({dt2:.2f}ms)")
            body2 = res2.json()
            print(f"Total Results: {body2.get('total_results')}")
            for r in body2.get("results", []):
                print(f"  - [{r['keyframe_id']}] Score: {r['score']}")
                for reason in r.get('reasons', []):
                    print(f"      * {reason}")
            assert res2.status_code == 200, "Count filter query failed!"
            print("✅ Test 3 PASSED!")

            print("\n--- Test 4: POST /search (Color Filters - Hungarian Matching) ---")
            payload3 = {
                "text_query": "presenter on stage with red backdrop",
                "mode": "hybrid",
                "color_filters": [
                    [255, 0, 0],
                    [0, 0, 255]
                ],
                "top_k": 3
            }
            t0 = time.time()
            res3 = client.post("/search", json=payload3)
            dt3 = (time.time() - t0) * 1000
            print(f"Status Code: {res3.status_code} ({dt3:.2f}ms)")
            body3 = res3.json()
            print(f"Total Results: {body3.get('total_results')}")
            for r in body3.get("results", []):
                print(f"  - [{r['keyframe_id']}] Score: {r['score']}")
                for reason in r.get('reasons', []):
                    print(f"      * {reason}")
            assert res3.status_code == 200, "Color filter query failed!"
            print("✅ Test 4 PASSED!")

            print("\n--- Test 5: POST /search/compare (Mode Comparison) ---")
            payload4 = {
                "text_query": "news anchor in studio",
                "top_k": 3
            }
            res4 = client.post("/search/compare", json=payload4)
            print(f"Status Code: {res4.status_code}")
            body4 = res4.json()
            print(f"Modes Compared: {list(body4.get('comparison', {}).keys())}")
            assert res4.status_code == 200, "Mode comparison failed!"
            print("✅ Test 5 PASSED!")

            print("\n" + "=" * 70)
            print("🎉 ALL API /search TESTS COMPLETED SUCCESSFULLY!")
            print("=" * 70)

if __name__ == "__main__":
    run_tests()
