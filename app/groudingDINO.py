from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from .schemas.reranking import (
    GroundingDetection,
    GroundingResult,
)



class GroundingDINO:
    """
    GroundingDINO Re-ranker.

    Pipeline:

        Query
          │
          ▼
        Top-K Images
          │
          ▼
        Batch GroundingDINO
          │
          ▼
        Grounding Scores
          │
          ▼
        Score Fusion
          │
          ▼
        Re-ranked Images

    Example input:

        images = [
            {
                "image_id": "frame_001",
                "image": PIL.Image,
                "retrieval_score": 0.82,
            },
            ...
        ]
    """

    def __init__(
        self,
         model_id: str = "IDEA-Research/grounding-dino-base",
        #model_id: str = "openmmlab-community/mm_grounding_dino_large_all",
        device: str | None = None,
        batch_size: int = 4,
        threshold: float = 0.25,
        text_threshold: float = 0.20,
        alpha: float = 0.4,
        beta: float = 0.6,
    ):
        """
        Args:
            model_id:
                HuggingFace GroundingDINO model.

            device:
                "cuda", "cuda:0", "cpu", ...

            batch_size:
                Number of images processed in one forward pass.

            threshold:
                Bounding box confidence threshold.

            text_threshold:
                Text matching threshold.

            alpha:
                Weight of original retrieval score.

            beta:
                Weight of GroundingDINO score.

            final_score =
                alpha * retrieval_score
                + beta * grounding_score
        """

        self.model_id = model_id

        self.device = (
            device
            if device is not None
            else (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.batch_size = batch_size

        self.threshold = threshold
        self.text_threshold = text_threshold

        self.alpha = alpha
        self.beta = beta

        # ==========================================
        # Processor
        # ==========================================

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_id
            )
        )

        # ==========================================
        # Model
        # ==========================================

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(
                self.model_id
            )
            .to(self.device)
        )

        self.model.eval()

        print(
            f"GroundingDINO loaded on {self.device}"
        )

    # ==================================================
    # Query
    # ==================================================

    @staticmethod
    def _prepare_query(
        query: str,
    ) -> str:

        query = query.strip()

        if not query.endswith("."):
            query += "."

        return query

    # ==================================================
    # Single Batch Inference
    # ==================================================

    @torch.inference_mode()
    def _infer_batch(
        self,
        images: list[Image.Image],
        query: str,
    ) -> list[dict]:
        """
        Run GroundingDINO on one batch.

        Args:
            images:
                List of PIL images.

            query:
                Same text query for the whole batch.

        Returns:
            List of detection results.
        """

        query = self._prepare_query(query)

        # ==========================================
        # Processor
        # ==========================================

        inputs = self.processor(
            images=images,
            text=[query] * len(images),
            return_tensors="pt",
            padding=True,
        )

        # ==========================================
        # Move tensors to GPU
        # ==========================================

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

        # ==========================================
        # Model inference
        # ==========================================

        outputs = self.model(
            **inputs
        )

        # ==========================================
        # Post processing
        # ==========================================

        target_sizes = [
            image.size[::-1]
            for image in images
        ]

        results = (
            self.processor
            .post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs["input_ids"],
                threshold=self.threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )
        )

        return results

    # ==================================================
    # Convert Detection Result
    # ==================================================

    @staticmethod
    def _parse_result(
        result: dict,
    ) -> tuple[
        float,
        list[GroundingDetection],
    ]:
        """
        Convert raw GroundingDINO result
        into grounding score + detections.
        """

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        detections = []

        # ==========================================
        # No detection
        # ==========================================

        if len(scores) == 0:
            return 0.0, detections

        # ==========================================
        # Grounding score
        # ==========================================

        grounding_score = float(
            scores.mean().item()
        )

        # ==========================================
        # Parse detections
        # ==========================================

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            detections.append(
                GroundingDetection(
                    label=label,
                    score=float(score),
                    bbox=[
                        float(value)
                        for value in box.tolist()
                    ],
                )
            )

        return grounding_score, detections

    # ==================================================
    # Score Fusion
    # ==================================================

    def _calculate_final_score(
        self,
        grounding_score: float,
        retrieval_score: float | None,
    ) -> float:

        # Nếu không có retrieval score
        # thì chỉ sử dụng GroundingDINO.

        if retrieval_score is None:
            return grounding_score

        return (
            self.alpha * retrieval_score
            + self.beta * grounding_score
        )

    # ==================================================
    # Re-rank
    # ==================================================

    def rerank(
        self,
        query: str,
        images: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[GroundingResult]:
        if not images:
            return []

        all_results = []

        # ==========================================
        # Process by batch
        # ==========================================

        for start in range(
            0,
            len(images),
            self.batch_size,
        ):

            batch_items = images[
                start:start + self.batch_size
            ]

            batch_images = [
                item["image"]
                for item in batch_items
            ]

            # ======================================
            # GroundingDINO batch inference
            # ======================================

            batch_results = self._infer_batch(
                images=batch_images,
                query=query,
            )

            # ======================================
            # Parse results
            # ======================================

            for item, result in zip(
                batch_items,
                batch_results,
            ):

                grounding_score, detections = (
                    self._parse_result(
                        result
                    )
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
                        keyframe_id=item["keyframe_id"],
                        grounding_score=grounding_score,
                        retrieval_score=retrieval_score,
                        final_score=final_score,
                        detections=detections,
                    )
                )

        # ==========================================
        # Sort
        # ==========================================

        all_results.sort(
            key=lambda result: result.final_score,
            reverse=True,
        )

        # ==========================================
        # Top-N
        # ==========================================

        if top_k is not None:
            all_results = all_results[:top_k]

        return all_results

    # ==================================================
    # Simple API
    # ==================================================

    def score(
        self,
        query: str,
        images: list[dict[str, Any]],
    ) -> list[GroundingResult]:

        """
        Alias cho rerank() nếu chỉ muốn scoring.
        """

        return self.rerank(
            query=query,
            images=images,
            top_k=None,
        )