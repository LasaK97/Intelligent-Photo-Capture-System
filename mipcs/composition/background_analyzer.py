import cv2
import numpy as np
import asyncio
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import colorsys
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BackgroundQuality(Enum):
    """Background quality ratings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    BUSY = "busy"
    POOR = "poor"


@dataclass
class DistractionElement:
    """Detected distraction in background"""
    location: Tuple[int, int]  # (x, y) center
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    severity: float  # 0-1
    distraction_type: str  # "bright_spot", "high_contrast", "motion", "clutter"
    confidence: float  # 0-1


@dataclass
class CleanArea:
    """Clean area suitable for subject placement"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]
    cleanliness_score: float  # 0-1
    size_score: float  # 0-1


@dataclass
class BackgroundAnalysis:
    """Complete background analysis result"""
    # Core metrics
    edge_density: float  # 0-1 (higher = busier)
    color_variance: float  # 0-1 (higher = more chaotic)
    distraction_score: float  # 0-1 (higher = more distractions)
    texture_uniformity: float  # 0-1 (higher = more uniform)

    # Color analysis
    dominant_colors: List[Tuple[int, int, int]]  # RGB colors
    color_harmony_score: float  # 0-1
    color_temperature: str  # "warm", "cool", "neutral"

    # Spatial analysis
    clean_areas: List[CleanArea]
    distraction_elements: List[DistractionElement]

    # Overall assessment
    quality_rating: BackgroundQuality
    overall_score: float  # 0-1 (higher = better for photography)
    recommendations: List[str]

    # Metadata
    processing_time_ms: float
    frame_dimensions: Tuple[int, int]  # (height, width)

class BackgroundAnalyzer:

    def __init__(self):
        settings = get_settings()

        # Config
        self.config = settings.auto_framing.composition.background
        self.edge_config = self.config.edge_detection
        self.color_config = self.config.color_analysis
        self.distraction_config = self.config.distraction_detection
        self.thresholds = self.config.thresholds

        # Cache
        self.cache_size = self.config.cache_size
        self.analysis_cache = {}

        # Performance tracking
        self.analysis_times = []

        logger.info(
            "background_analyzer_initialized",
            canny_thresholds=f"{self.edge_config.threshold_low}-{self.edge_config.threshold_high}",
            dominant_colors=self.color_config.dominant_colors_count,
            cache_size=self.cache_size
        )

    async def analyze_background(
            self,
            frame: np.ndarray,
            subject_mask: Optional[np.ndarray] = None,
            region_of_interest: Optional[Tuple[int, int, int, int]] = None
    ) -> BackgroundAnalysis:
        """analyze background quality. """
        start_time = time.time()

        # check cache
        frame_hash = self._hash_frame(frame)
        if frame_hash in self.analysis_cache:
            logger.debug("background_analysis_cache_hit")
            return self.analysis_cache[frame_hash]

        # extract frame info
        height, width = frame.shape[:2]

        # extract background (exclude subjects)
        background = self._extract_background(frame, subject_mask, region_of_interest)

        # analyze edge density
        edge_density = await self._analyze_edge_density(background)

        # analyze color harmony
        color_analysis = await self._analyze_color_harmony(background)

        # detect distractions
        distractions = await self._detect_distractions(background)
        distraction_score = self._calculate_distraction_score(distractions)

        # analyze texture
        texture_uniformity = await self._analyze_texture(background)

        # find clean areas
        clean_areas = await self._find_clean_areas(background)

        # overall assessment
        quality_rating, overall_score = self._assess_quality(
            edge_density,
            color_analysis['variance'],
            distraction_score,
            texture_uniformity
        )

        # generate recommendations
        recommendations = self._generate_recommendations(
            edge_density,
            color_analysis,
            distractions,
            quality_rating
        )

        # build result
        analysis = BackgroundAnalysis(
            edge_density=edge_density,
            color_variance=color_analysis['variance'],
            distraction_score=distraction_score,
            texture_uniformity=texture_uniformity,
            dominant_colors=color_analysis['dominant_colors'],
            color_harmony_score=color_analysis['harmony_score'],
            color_temperature=color_analysis['temperature'],
            clean_areas=clean_areas,
            distraction_elements=distractions,
            quality_rating=quality_rating,
            overall_score=overall_score,
            recommendations=recommendations,
            processing_time_ms=(time.time() - start_time) * 1000,
            frame_dimensions=(height, width)
        )

        # cache result
        self._cache_result(frame_hash, analysis)

        # track performance
        self.analysis_times.append(time.time() - start_time)

        logger.debug(
            "background_analyzed",
            edge_density=f"{edge_density:.2f}",
            distraction_score=f"{distraction_score:.2f}",
            quality=quality_rating.value,
            time_ms=f"{analysis.processing_time_ms:.1f}"
        )

        return analysis

    def _extract_background(
            self,
            frame: np.ndarray,
            subject_mask: Optional[np.ndarray],
            roi: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """extract background region (exclude subjects)"""

        # apply ROI if specified
        if roi:
            x, y, w, h = roi
            background = frame[y:y + h, x:x + w].copy()
        else:
            background = frame.copy()

        # exclude subjects if mask provided
        if subject_mask is not None:
            if roi:
                x, y, w, h = roi
                mask_roi = subject_mask[y:y + h, x:x + w]
            else:
                mask_roi = subject_mask

            # zero out subject areas
            background = cv2.bitwise_and(background, background, mask=~mask_roi)

        return background

    async def _analyze_edge_density(self, background: np.ndarray) -> float:
        """analyze edge density using Canny edge detection.
        High edge density = busy background.
        """
        # convert to grayscale
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # apply Gaussian blur
        blur_size = self.edge_config.blur_kernel_size
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        # canny edge detection
        edges = cv2.Canny(
            blurred,
            self.edge_config.canny_low,
            self.edge_config.canny_high
        )

        # calculate density (percentage of edge pixels)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        density = edge_pixels / total_pixels

        return float(density)

    async def _analyze_color_harmony(self, background: np.ndarray) -> Dict:
        """
        analyze color harmony using K-means clustering in HSV space.
        returns dominant colors and harmony score.
        """
        # convert to HSV for better color perception
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

        # reshape for K-means
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # remove black pixels (masked areas)
        non_black = pixels[np.sum(pixels, axis=1) > 10]

        if len(non_black) < 10:
            return {
                'variance': 0.0,
                'dominant_colors': [],
                'harmony_score': 0.5,
                'temperature': 'neutral'
            }

        # K-means clustering
        k = min(self.color_config.num_clusters, len(non_black))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            non_black,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS
        )

        # convert centers back to RGB
        centers_rgb = []
        for center in centers:
            hsv_color = np.uint8([[center]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            centers_rgb.append(tuple(map(int, rgb_color)))

        # calculate color variance (diversity)
        unique, counts = np.unique(labels, return_counts=True)
        distribution = counts / len(labels)
        variance = float(np.std(distribution))

        # calculate color harmony
        harmony_score = self._calculate_color_harmony(centers)

        # cetermine color temperature
        temperature = self._determine_color_temperature(centers)

        # get top N dominant colors
        max_colors = self.color_config.max_dominant_colors
        top_indices = np.argsort(counts)[-max_colors:][::-1]
        dominant_colors = [centers_rgb[i] for i in top_indices]

        return {
            'variance': variance,
            'dominant_colors': dominant_colors,
            'harmony_score': harmony_score,
            'temperature': temperature
        }

    def _calculate_color_harmony(self, hsv_centers: np.ndarray) -> float:
        """
        calculate color harmony score.
        Complementary colors = high harmony.
        """
        if len(hsv_centers) < 2:
            return 0.5

        harmony_scores = []

        # check pairs for complementary colors
        for i in range(len(hsv_centers)):
            for j in range(i + 1, len(hsv_centers)):
                h1 = hsv_centers[i][0]
                h2 = hsv_centers[j][0]

                # complementary: ~180 degrees apart on hue wheel
                hue_diff = abs(h1 - h2)
                if hue_diff > 90:
                    hue_diff = 180 - hue_diff

                # score based on proximity to complementary
                complement_score = 1.0 - abs(hue_diff - 90) / 90
                harmony_scores.append(complement_score)

        return float(np.mean(harmony_scores)) if harmony_scores else 0.5

    def _determine_color_temperature(self, hsv_centers: np.ndarray) -> str:
        """determine overall color temperature (warm/cool/neutral)"""
        if len(hsv_centers) == 0:
            return "neutral"

        # hue ranges: 0-30 = warm red, 30-90 = warm yellow, 90-150 = cool green, 150-180 = cool blue
        hues = hsv_centers[:, 0]

        warm_count = np.sum((hues < 60) | (hues > 150))
        cool_count = np.sum((hues >= 60) & (hues <= 150))

        if warm_count > cool_count * 1.5:
            return "warm"
        elif cool_count > warm_count * 1.5:
            return "cool"
        else:
            return "neutral"

    async def _detect_distractions(self, background: np.ndarray) -> List[DistractionElement]:
        """
        Detect distracting elements in background.
        Types: bright spots, high contrast areas, cluttered regions.
        """
        distractions = []

        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # 1. detect bright spots
        _, bright_mask = cv2.threshold(
            gray,
            self.distraction_config.brightness_variance_threshold,
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum size
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                distractions.append(DistractionElement(
                    location=center,
                    bbox=(x, y, w, h),
                    severity=0.7,
                    distraction_type="bright_spot",
                    confidence=0.8
                ))

        # 2. detect high-frequency regions (clutter)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_abs = np.abs(laplacian)

        # find regions with high variance
        block_size = 32
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = laplacian_abs[y:y + block_size, x:x + block_size]
                variance = np.var(block)

                if variance > self.distraction_config.high_frequency_threshold:
                    center = (x + block_size // 2, y + block_size // 2)

                    distractions.append(DistractionElement(
                        location=center,
                        bbox=(x, y, block_size, block_size),
                        severity=min(1.0, variance / 200),
                        distraction_type="clutter",
                        confidence=0.7
                    ))

        # limit to top distractions
        distractions.sort(key=lambda d: d.severity, reverse=True)
        return distractions[:10]

    def _calculate_distraction_score(self, distractions: List[DistractionElement]) -> float:
        """calculate overall distraction score from detected elements"""
        if not distractions:
            return 0.0

        # weighted by severity and confidence
        total_score = sum(d.severity * d.confidence for d in distractions)

        # normalize by number of distractions (more = worse)
        normalized = total_score / max(1, len(distractions)) * min(1.0, len(distractions) / 5)

        return min(1.0, normalized)

    async def _analyze_texture(self, background: np.ndarray) -> float:
        """
        Analyze texture uniformity.
        High uniformity = clean, simple background.
        """
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # calculate local variance
        mean = cv2.blur(gray, (15, 15))
        sqr_mean = cv2.blur(gray ** 2, (15, 15))
        variance = sqr_mean - mean ** 2

        # Uniformity = inverse of variance
        avg_variance = np.mean(variance)
        uniformity = 1.0 - min(1.0, avg_variance / 1000)

        return float(uniformity)

    async def _find_clean_areas(self, background: np.ndarray) -> List[CleanArea]:
        """
        find clean areas suitable for subject placement.
        Clean = low edge density + uniform color.
        """
        gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # compute local complexity
        edges = cv2.Canny(gray, 50, 150)

        clean_areas = []
        block_size = 64

        for y in range(0, height - block_size, block_size // 2):
            for x in range(0, width - block_size, block_size // 2):
                block_edges = edges[y:y + block_size, x:x + block_size]
                block_gray = gray[y:y + block_size, x:x + block_size]

                # calculate cleanliness
                edge_density = np.count_nonzero(block_edges) / block_edges.size
                color_variance = np.std(block_gray) / 255

                cleanliness = 1.0 - (edge_density * 0.7 + color_variance * 0.3)

                if cleanliness > 0.7:  # high cleanliness threshold
                    center = (x + block_size // 2, y + block_size // 2)
                    size_score = min(1.0, (block_size * block_size) / (height * width * 0.1))

                    clean_areas.append(CleanArea(
                        bbox=(x, y, block_size, block_size),
                        center=center,
                        cleanliness_score=cleanliness,
                        size_score=size_score
                    ))

        # sort by cleanliness and return top areas
        clean_areas.sort(key=lambda a: a.cleanliness_score, reverse=True)
        return clean_areas[:5]

    def _assess_quality(
            self,
            edge_density: float,
            color_variance: float,
            distraction_score: float,
            texture_uniformity: float
    ) -> Tuple[BackgroundQuality, float]:
        """assess overall background quality."""
        # calculate weighted score
        score = (
                (1.0 - edge_density) * 0.3 +
                (1.0 - color_variance) * 0.2 +
                (1.0 - distraction_score) * 0.3 +
                texture_uniformity * 0.2
        )

        # determine rating
        if score >= 0.85:
            rating = BackgroundQuality.EXCELLENT
        elif score >= 0.70:
            rating = BackgroundQuality.GOOD
        elif score >= 0.55:
            rating = BackgroundQuality.ACCEPTABLE
        elif score >= 0.40:
            rating = BackgroundQuality.BUSY
        else:
            rating = BackgroundQuality.POOR

        return rating, score

    def _generate_recommendations(
            self,
            edge_density: float,
            color_analysis: Dict,
            distractions: List[DistractionElement],
            quality: BackgroundQuality
    ) -> List[str]:
        """generate actionable recommendations"""
        recommendations = []

        if edge_density > self.thresholds.max_edge_density:
            recommendations.append("Background is too busy - consider simpler backdrop or shallow depth of field")

        if color_analysis['variance'] > self.thresholds.max_color_variance:
            recommendations.append("Colors are too varied - consider more uniform background")

        if len(distractions) > 3:
            recommendations.append(f"{len(distractions)} distracting elements detected - reframe or adjust angle")

        if quality == BackgroundQuality.POOR:
            recommendations.append("Poor background quality - strongly recommend changing location or angle")

        if not recommendations:
            recommendations.append("Background quality is good for photography")

        return recommendations[:5]

    def _hash_frame(self, frame: np.ndarray) -> int:
        """create hash of frame for caching"""
        # simple hash based on mean values (fast but approximate)
        return hash((frame.mean(), frame.shape[0], frame.shape[1]))

    def _cache_result(self, frame_hash: int, analysis: BackgroundAnalysis):
        """cache analysis result"""
        self.analysis_cache[frame_hash] = analysis

        # limit cache size
        if len(self.analysis_cache) > self.cache_size:
            # Remove oldest (first inserted)
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.analysis_times:
            return {}

        return {
            'avg_time_ms': np.mean(self.analysis_times) * 1000,
            'max_time_ms': np.max(self.analysis_times) * 1000,
            'total_analyses': len(self.analysis_times),
            'cache_hits': len(self.analysis_cache),
            'cache_size': self.cache_size
        }

    # TODO: MANRIIX V2 --> incorporate sensors instead of estimations and analyzes use real data