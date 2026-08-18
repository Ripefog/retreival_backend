#!/usr/bin/env python3
"""Send a POST /search/rerank request with 100 real keyframe IDs from keyframes/L21/L21_V001.

Usage:
    python test_rerank_api.py
    python test_rerank_api.py --url http://127.0.0.1:8000/search/rerank --count 100
    python test_rerank_api.py --folder keyframes/L21/L21_V001 --query "person with cup" --top_k 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import requests


DEFAULT_FOLDER = Path("keyframes/L21/L21_V001")
DEFAULT_URL = "http://127.0.0.1:8000/search/rerank"


def build_payload(folder: str, count: int, query: str, top_k: int) -> dict:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    files = sorted(
        p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(files) < count:
        raise ValueError(
            f"Requested {count} frames but folder only has {len(files)} image files: {root}"
        )

    selected = files[:count]
    frames = [
        {
            "keyframe_id": name,
            "retrieval_score": 0.0,
        }
        for name in selected
    ]

    return {
        "query": query,
        "top_k": top_k,
        "frames": frames,
    }


def visualize_rerank_response(response_json: dict, image_root: str, output_dir: str = "rerank_visualizations", max_items: int = 5):
    """Try to draw boxes for returned results if matching image files exist; visualize the highest final_score results first."""
    results = response_json.get("results", [])
    if not results:
        print("[CHECK] Response has no results -> visualization not created.")
        return []

    results = sorted(
        results,
        key=lambda x: float(x.get("final_score", float("-inf"))),
        reverse=True,
    )

    root = Path(image_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for result in results[:max_items]:
        keyframe_id = result.get("keyframe_id")
        if not keyframe_id:
            continue

        image_path = root / keyframe_id
        if not image_path.exists():
            print(f"[CHECK] Skip visualization for {keyframe_id}: file not found in {root}")
            continue

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for det in result.get("detections", []):
            bbox = det.get("bbox") or []
            if len(bbox) < 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            draw.text((max(0, x0), max(0, y0 - 18)), f"{det.get('label', 'obj')}", fill="red")

        save_path = out_dir / f"{Path(keyframe_id).stem}_boxed.jpg"
        image.save(save_path)
        saved.append(save_path)
        print(f"[CHECK] Visualized: {save_path} | final_score={result.get('final_score')}")

    if not saved:
        print("[CHECK] Response received but no visualization was created because matching keyframe images were not found.")
    else:
        print(f"[CHECK] Visualization completed for {len(saved)} highest-scoring result(s).")

    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a rerank request with real keyframe IDs from L21_V001")
    parser.add_argument("--folder", default=str(DEFAULT_FOLDER), help="Folder containing keyframe images")
    parser.add_argument("--url", default=DEFAULT_URL, help="Endpoint URL for POST /search/rerank")
    parser.add_argument("--count", type=int, default=478, help="Number of keyframes to send")
    parser.add_argument("--query", default="A man wearing a blue shirt is riding a motorcycle", help="Query for reranking")
    parser.add_argument("--top_k", type=int, default=100, help="Number of results to return")
    parser.add_argument("--visualize", action="store_true", help="Try to visualize result boxes when matching images exist")
    parser.add_argument("--visualize-max", type=int, default=5, help="Maximum number of result images to visualize")
    args = parser.parse_args()

    try:
        payload = build_payload(args.folder, args.count, args.query, args.top_k)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"Sending {len(payload['frames'])} frames to {args.url}")
    print(f"Query: {payload['query']}")
    print(f"First frame: {payload['frames'][0]['keyframe_id']}")
    print(f"Last frame: {payload['frames'][-1]['keyframe_id']}")

    try:
        response = requests.post(args.url, json=payload, timeout=300)
        print(f"Status: {response.status_code}")
        print(response.text[:4000])

        if response.ok and args.visualize:
            try:
                response_json = response.json()
                print("[CHECK] Checking visualization status after response...")
                visualize_rerank_response(response_json, args.folder, max_items=args.visualize_max)
            except Exception as exc:
                print(f"[CHECK] Visualization check failed: {exc}")

        if response.ok:
            print("[CHECK] Response received successfully. Visualization check finished.")
        return 0 if response.ok else 1
    except requests.RequestException as exc:
        print(f"[ERROR] Request failed: {exc}")
        print("Make sure the API server is running, for example:")
        print("  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
