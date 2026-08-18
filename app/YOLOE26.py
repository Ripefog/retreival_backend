from __future__ import annotations

import os
import re
from typing import Any, List, Union

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

from .config import settings

from .schemas.reranking import (
    GroundingDetection,
    GroundingResult,
)


ImageInput = Union[str, Image.Image]


class YOLOE26LReranker:

    def __init__(
        self,
        repo_id: str = "openvision/yoloe26-l-seg",
        filename: str = "model.pt",
        device: Union[int, str] = 0,
        conf: float = 0.20,
        iou: float = 0.50,
        imgsz: int = 640,
        batch_size: int = 16,
        alpha: float = 0.4,
        beta: float = 0.6,
    ):
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        print(
            f"Downloading YOLOE model "
            f"{repo_id}/{filename}..."
        )

        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )

        print(
            f"Loading YOLOE model from "
            f"{model_path}..."
        )

        self.model = YOLO(model_path)
        self.classes: List[str] = []

        print(
            f"YOLOE-26L loaded successfully "
            f"on device={self.device}"
        )

    @staticmethod
    def _normalize_class_name(value: str) -> str:
        if not value:
            return ""
        cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", value).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _fallback_extract_classes(self, query: str) -> List[str]:
        text = query.lower()
        tokens = re.findall(r"[a-zA-Z0-9]+", text)
        if not tokens:
            return []

        common_objects = {
            "person": ["person", "people", "man", "woman", "child", "human", "girl", "boy"],
            "bicycle": ["bicycle", "bike", "bicyclist"],
            "car": ["car", "truck", "vehicle", "motorcycle", "motorbike", "bus"],
            "dog": ["dog", "puppy"],
            "cat": ["cat", "kitten"],
            "table": ["table", "desk"],
            "chair": ["chair"],
            "cup": ["cup", "mug", "glass"],
            "food": ["food", "dish", "plate", "pizza", "burger"],
            "cell phone": ["cell phone", "phone", "smartphone", "mobile phone"],
            "book": ["book", "notebook"],
            "laptop": ["laptop", "computer", "notebook computer"],
            "bottle": ["bottle", "water bottle"],
            "backpack": ["backpack", "bag"],
            "shoe": ["shoe", "sneaker", "boots"],
            "tree": ["tree"],
            "building": ["building", "house", "tower"],
            "traffic light": ["traffic light", "signal"],
            "bench": ["bench"],
            "skateboard": ["skateboard"],
            "surfboard": ["surfboard"],
            "train": ["train"],
            "boat": ["boat"],
            "airplane": ["airplane", "plane"],
            "basketball": ["basketball"],
            "baseball bat": ["baseball bat", "bat"],
            "tennis racket": ["tennis racket", "racket"],
            "snowboard": ["snowboard"],
            "kite": ["kite"],
            "umbrella": ["umbrella"],
            "frisbee": ["frisbee"],
        }

        detected: List[str] = []
        for canonical, aliases in common_objects.items():
            for alias in aliases:
                if alias in text:
                    detected.append(canonical)
                    break

        if not detected:
            phrase = " ".join(tokens[:10])
            if phrase:
                detected = [self._normalize_class_name(phrase)]

        return self._deduplicate_classes(detected)

    @staticmethod
    def _deduplicate_classes(classes: List[str]) -> List[str]:
        seen = set()
        clean: List[str] = []
        for item in classes:
            value = item.strip()
            if not value:
                continue
            normalized = " ".join(value.split()).lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            clean.append(value)
        return clean
    
    def extract_classes_from_query(
        self,
        query: str,
    ) -> List[str]:

        if not query or not str(query).strip():
            raise ValueError("query cannot be empty.")

        cleaned_query = str(query).strip()

        api_key = (
            settings.GEMINI_API_KEY
            or settings.GOOGLE_API_KEY
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        if api_key:
            try:
                import json
                from google import genai

                client = genai.Client(
                    api_key=api_key
                )

                prompt = f"""
    You are an expert in open-vocabulary object detection
    and YOLOE-26L.

    Your task is to extract the most useful visual object
    classes from a natural-language video retrieval query.

    QUERY:
    {cleaned_query}

    The output will be passed directly to YOLOE-26L
    as open-vocabulary text prompts.

    RULES:

    1. Extract only visually detectable objects.

    2. Extract people when the query mentions:
    man, woman, boy, girl, child, person, people,
    cyclist, pedestrian, etc.
    
    Normalize them to:
    "person"

    3. Extract physical objects:
    bicycle, car, bus, motorcycle, truck,
    dog, cat, bag, phone, laptop, etc.

    4. Preserve important visual attributes when useful.

    Examples:
    "red shirt"
    "blue car"
    "black bicycle"
    "white dog"

    5. Do NOT return actions as classes.

    Bad:
    "riding"
    "walking"
    "running"
    "holding"
    "sitting"

    6. Do NOT return locations.

    Bad:
    "street"
    "road"
    "park"
    "classroom"

    7. Do NOT return abstract concepts.

    Bad:
    "happy"
    "dangerous"
    "beautiful"
    "crowded"

    8. Prefer short, concrete English phrases.

    9. Avoid long descriptions.

    Bad:
    "person wearing a red shirt and riding a bicycle"

    Good:
    "person"
    "red shirt"
    "bicycle"

    10. Keep the number of classes between 1 and 6.

    11. Remove duplicates.

    12. Do not invent objects that are not explicitly
        mentioned or strongly implied by the query.

    13. If a person is described using a role such as:
        cyclist, rider, pedestrian, shopper, worker,
        normalize it to "person".

    14. If an object has an important color attribute,
        preserve it.

    15. Return ONLY JSON.

    OUTPUT FORMAT:

    {{
        "classes": [
            "person",
            "bicycle",
            "red shirt"
        ]
    }}
    """

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                )

                text = getattr(
                    response,
                    "text",
                    "",
                ) or ""

                if not text:
                    raise ValueError(
                        "Gemini returned empty response"
                    )

                data = json.loads(text)

                items = data.get(
                    "classes",
                    [],
                )

                if not isinstance(items, list):
                    raise ValueError(
                        "Gemini 'classes' must be a list"
                    )

                classes = self._deduplicate_classes(
                    [
                        self._normalize_class_name(
                            str(item)
                        )
                        for item in items
                        if str(item).strip()
                    ]
                )

                if classes:
                    print(
                        f"Gemini extracted classes: "
                        f"{classes}"
                    )

                    return classes

            except Exception as exc:
                print(
                    "Gemini class extraction failed, "
                    f"falling back: {exc}"
                )

        detected = self._fallback_extract_classes(
            cleaned_query
        )

        if detected:
            return detected

        return [
            self._normalize_class_name(
                cleaned_query
            )
        ][:1]

    def set_classes(
        self,
        classes: List[str],
    ) -> None:

        classes = self._deduplicate_classes([
            self._normalize_class_name(c)
            for c in classes
            if c and c.strip()
        ])

        if not classes:
            raise ValueError(
                "classes cannot be empty."
            )

        text_embeddings = (
            self.model.get_text_pe(classes)
        )

        self.model.set_classes(
            classes,
            text_embeddings,
        )

        self.classes = classes

        print(
            f"YOLOE classes set to: "
            f"{self.classes}"
        )

    @staticmethod
    def _load_image(
        image_input: ImageInput,
    ) -> Image.Image:

        if isinstance(
            image_input,
            Image.Image,
        ):
            return image_input.convert("RGB")

        if image_input.startswith(
            ("http://", "https://")
        ):
            response = requests.get(
                image_input,
                stream=True,
                timeout=10,
            )

            response.raise_for_status()

            return Image.open(
                response.raw
            ).convert("RGB")

        if os.path.exists(image_input):
            return Image.open(
                image_input
            ).convert("RGB")

        raise FileNotFoundError(
            f"Image not found: {image_input}"
        )

    @torch.inference_mode()
    def _infer_batch(
        self,
        images: List[Image.Image],
    ):
        return self.model.predict(
            source=images,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    def _parse_result(
            self,
            result,
        ) -> tuple[
            float,
            List[GroundingDetection],
        ]:

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            return 0.0, []

        detections = []

        best_scores = {
            class_name: 0.0
            for class_name in self.classes
        }

        for i in range(
            len(result.boxes)
        ):
            class_id = int(
                result.boxes.cls[i].item()
            )

            confidence = float(
                result.boxes.conf[i].item()
            )

            bbox = (
                result.boxes.xyxy[i]
                .tolist()
            )

            if (
                0 <= class_id
                < len(self.classes)
            ):
                label = self.classes[class_id]
            else:
                label = str(class_id)

            detections.append(
                GroundingDetection(
                    label=label,
                    score=confidence,
                    bbox=[
                        float(v)
                        for v in bbox
                    ],
                )
            )

            if label in best_scores:
                best_scores[label] = max(
                    best_scores[label],
                    confidence,
                )

        grounding_score = (
            sum(best_scores.values())
            / len(best_scores)
            if best_scores
            else 0.0
        )

        return (
            grounding_score,
            detections,
        )

    def _calculate_final_score(
        self,
        grounding_score: float,
        retrieval_score: float | None,
    ) -> float:

        if retrieval_score is None:
            return grounding_score

        return (
            self.alpha * retrieval_score
            + self.beta * grounding_score
        )

    def rerank(
        self,
        query_classes: List[str],
        images: List[dict[str, Any]],
        top_k: int | None = None,
    ) -> List[GroundingResult]:

        if not images:
            return []

        self.set_classes(query_classes)

        all_results = []

        for start in range(
            0,
            len(images),
            self.batch_size,
        ):
            end = min(
                start + self.batch_size,
                len(images),
            )

            batch_items = images[
                start:end
            ]

            batch_images = [
                self._load_image(
                    item["image"]
                )
                for item in batch_items
            ]

            print(
                f"YOLOE inference "
                f"{start + 1}-{end}/"
                f"{len(images)}"
            )

            batch_results = self._infer_batch(
                batch_images
            )

            for item, result in zip(
                batch_items,
                batch_results,
            ):
                (
                    grounding_score,
                    detections,
                ) = self._parse_result(
                    result
                )

                retrieval_score = item.get(
                    "retrieval_score"
                )

                final_score = (
                    self._calculate_final_score(
                        grounding_score=grounding_score,
                        retrieval_score=retrieval_score,
                    )
                )

                all_results.append(
                    GroundingResult(
                        keyframe_id=item[
                            "keyframe_id"
                        ],
                        grounding_score=grounding_score,
                        retrieval_score=retrieval_score,
                        final_score=final_score,
                        detections=detections,
                    )
                )

        all_results.sort(
            key=lambda x: x.final_score,
            reverse=True,
        )

        if top_k is not None:
            return all_results[:top_k]

        return all_results

    def score(
        self,
        query_classes: List[str],
        images: List[dict[str, Any]],
    ) -> List[GroundingResult]:

        return self.rerank(
            query_classes=query_classes,
            images=images,
            top_k=None,
        )