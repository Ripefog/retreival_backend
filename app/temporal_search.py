
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from itertools import product
import numpy as np
from collections import defaultdict

from .models import (
    TemporalQuery, TemporalSearchRequest, TemporalSearchResponse,
    TemporalStep, TemporalSequence
)

logger = logging.getLogger(__name__)

class TemporalSearchEngine:
    """
    Engine để thực hiện temporal search - tìm kiếm chuỗi hành động theo thứ tự thời gian
    """
    
    def __init__(self, hybrid_retriever):
        """
        Args:
            hybrid_retriever: Instance của HybridRetriever để thực hiện search cơ bản
        """
        self.hybrid_retriever = hybrid_retriever
        
    async def temporal_search(self, request: TemporalSearchRequest) -> TemporalSearchResponse:
        """
        Thực hiện temporal search theo request
        
        Args:
            request: TemporalSearchRequest với các query tuần tự
            
        Returns:
            TemporalSearchResponse với các sequences được tìm thấy
        """
        start_time = time.time()
        
        # Validate input
        if len(request.sequential_queries) < 2:
            raise ValueError("Temporal search requires at least 2 sequential queries")
            
        # Sort queries by step to ensure correct order
        sorted_queries = sorted(request.sequential_queries, key=lambda x: x.step)
        
        logger.info(f"Starting temporal search with {len(sorted_queries)} sequential queries")
        
        # Step 1: Execute individual queries in parallel
        query_results = await self._execute_sequential_queries(sorted_queries, request)
        
        # Step 2: Generate candidate sequences with temporal constraints
        candidate_sequences = self._generate_candidate_sequences(
            query_results, sorted_queries, request
        )
        
        # Step 3: Score and rank sequences
        scored_sequences = self._score_sequences(candidate_sequences, request)
        
        # Step 4: Select top sequences
        top_sequences = scored_sequences[:request.top_sequences]
        
        processing_time = time.time() - start_time
        
        logger.info(f"Temporal search completed: {len(candidate_sequences)} candidates -> {len(top_sequences)} final sequences in {processing_time:.2f}s")
        
        return TemporalSearchResponse(
            sequential_queries=sorted_queries,
            query_results=query_results,
            temporal_sequences=top_sequences,
            total_sequences=len(candidate_sequences),
            processing_time=processing_time
        )
    
    async def _execute_sequential_queries(
        self, 
        queries: List[TemporalQuery], 
        request: TemporalSearchRequest
    ) -> List[List[Dict[str, Any]]]:
        """
        Thực hiện các query tuần tự song song
        """
        tasks = []
        for query in queries:
            task = self.hybrid_retriever.search(
                text_query=query.text_query,
                mode=request.mode.value,
                user_query="",  # No user filtering for temporal search
                object_filters=request.object_filters,
                color_filters=request.color_filters,
                ocr_query=request.ocr_query,
                asr_query=request.asr_query,
                top_k=request.top_k_per_query
            )
            tasks.append(task)
        
        # Execute all queries in parallel
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Executed {len(queries)} queries, got results: {[len(r) for r in results]}")
        return results
    
    def _generate_candidate_sequences(
        self,
        query_results: List[List[Dict[str, Any]]],
        queries: List[TemporalQuery],
        request: TemporalSearchRequest
    ) -> List[Dict[str, Any]]:
        """
        Tạo candidate sequences từ kết quả các query riêng lẻ
        """
        # Group results by video_id for each query
        video_grouped_results = []
        for results in query_results:
            video_group = defaultdict(list)
            for result in results:
                video_id = result.get('video_id', '')
                if video_id:
                    video_group[video_id].append(result)
            video_grouped_results.append(video_group)
        
        # Find common videos across all queries
        if not video_grouped_results:
            return []
        
        common_videos = set(video_grouped_results[0].keys())
        for video_group in video_grouped_results[1:]:
            common_videos &= set(video_group.keys())
        
        logger.info(f"Found {len(common_videos)} videos with results for all queries")
        
        # Generate sequences for each common video
        candidate_sequences = []
        sequence_id = 1
        
        for video_id in common_videos:
            # Get results for this video from all queries
            video_results = [
                video_grouped_results[i][video_id] 
                for i in range(len(queries))
            ]
            
            # Generate all possible combinations
            for combination in product(*video_results):
                # Check temporal constraints
                if self._validate_temporal_constraints(combination, request):
                    sequence = {
                        'sequence_id': sequence_id,
                        'video_id': video_id,
                        'combination': combination,
                        'queries': queries
                    }
                    candidate_sequences.append(sequence)
                    sequence_id += 1
        
        logger.info(f"Generated {len(candidate_sequences)} candidate sequences")
        return candidate_sequences
    
    def _validate_temporal_constraints(
        self,
        combination: Tuple[Dict[str, Any], ...],
        request: TemporalSearchRequest
    ) -> bool:
        """
        Kiểm tra ràng buộc thời gian cho một combination
        """
        timestamps = [result.get('timestamp', 0.0) for result in combination]
        
        # Check if timestamps are in ascending order
        for i in range(1, len(timestamps)):
            time_gap = timestamps[i] - timestamps[i-1]
            
            # Must be in temporal order
            if time_gap < request.min_time_gap:
                return False
            
            # Must not exceed max time gap
            if time_gap > request.max_time_gap:
                return False
        
        return True
    
    def _score_sequences(
        self,
        candidate_sequences: List[Dict[str, Any]],
        request: TemporalSearchRequest
    ) -> List[TemporalSequence]:
        """
        Tính điểm và xếp hạng các sequences
        """
        scored_sequences = []
        
        for seq_data in candidate_sequences:
            combination = seq_data['combination']
            queries = seq_data['queries']
            
            # Calculate sequence score components
            semantic_scores = [result.get('score', 0.0) for result in combination]
            avg_semantic_score = np.mean(semantic_scores)
            
            # Calculate temporal consistency score
            timestamps = [result.get('timestamp', 0.0) for result in combination]
            time_gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            
            # Temporal consistency: prefer consistent gaps, penalize too long/short gaps
            temporal_consistency = self._calculate_temporal_consistency(time_gaps, request)
            
            # Final sequence score (weighted combination)
            sequence_score = (
                0.6 * avg_semantic_score +
                0.3 * temporal_consistency +
                0.1 * min(semantic_scores)  # Boost sequences where all steps are strong
            )
            
            # Create TemporalStep objects
            steps = []
            for i, (result, query) in enumerate(zip(combination, queries)):
                step = TemporalStep(
                    step=query.step,
                    keyframe_id=result.get('keyframe_id', ''),
                    video_id=result.get('video_id', ''),
                    timestamp=result.get('timestamp', 0.0),
                    score=result.get('score', 0.0),
                    text_query=query.text_query
                )
                steps.append(step)
            
            # Create TemporalSequence
            sequence = TemporalSequence(
                sequence_id=seq_data['sequence_id'],
                video_id=seq_data['video_id'],
                steps=steps,
                sequence_score=round(sequence_score, 4),
                temporal_consistency=round(temporal_consistency, 4),
                time_gaps=time_gaps,
                total_duration=round(timestamps[-1] - timestamps[0], 2)
            )
            
            scored_sequences.append(sequence)
        
        # Sort by sequence score
        scored_sequences.sort(key=lambda x: x.sequence_score, reverse=True)
        
        return scored_sequences
    
    def _calculate_temporal_consistency(
        self,
        time_gaps: List[float],
        request: TemporalSearchRequest
    ) -> float:
        """
        Tính điểm nhất quán về thời gian
        """
        if not time_gaps:
            return 1.0
            
        # Ideal time gap (somewhere in the middle of min/max range)
        ideal_gap = (request.min_time_gap + request.max_time_gap) / 2
        
        # Calculate how close gaps are to ideal
        gap_scores = []
        for gap in time_gaps:
            # Normalize gap score (closer to ideal = higher score)
            if gap <= ideal_gap:
                # Below ideal: scale from min_time_gap to ideal_gap
                score = (gap - request.min_time_gap) / (ideal_gap - request.min_time_gap)
            else:
                # Above ideal: scale from ideal_gap to max_time_gap  
                score = 1.0 - (gap - ideal_gap) / (request.max_time_gap - ideal_gap)
            
            gap_scores.append(max(0.0, min(1.0, score)))
        
        # Return average consistency score
        return np.mean(gap_scores)

