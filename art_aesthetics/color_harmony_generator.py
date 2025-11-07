"""
Main orchestrator class for the AI Color Harmony Generator
"""

import logging
from typing import Optional, List
from pathlib import Path

from models import (
    Color,
    ColorPalette,
    ArtistStyle,
    Emotion,
    ColorHarmonyType,
    ColorExtractor,
    EmbeddingService,
    PaletteGenerator
)
from services import (
    OpenCVColorExtractor,
    CLIPEmbeddingService,
    ColorHarmonyGenerator
)

logger = logging.getLogger(__name__)


class ColorHarmonyManager:
    """High-level manager for color palette generation"""
    
    def __init__(
        self,
        color_extractor: Optional[ColorExtractor] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """Initialize the manager with services"""
        self.color_extractor = color_extractor or OpenCVColorExtractor()
        self.embedding_service = embedding_service or CLIPEmbeddingService()
        self.generator = ColorHarmonyGenerator(
            self.color_extractor,
            self.embedding_service
        )
    
    def generate_from_artist_style(
        self,
        artist_style: ArtistStyle,
        num_colors: int = 5,
        emotion: Optional[Emotion] = None
    ) -> ColorPalette:
        """Generate a palette inspired by a specific artist style"""
        logger.info(f"Generating palette for {artist_style.value} style")
        return self.generator.generate_palette(
            artist_style=artist_style,
            emotion=emotion,
            num_colors=num_colors
        )
    
    def generate_from_emotion(
        self,
        emotion: Emotion,
        num_colors: int = 5,
        base_color: Optional[Color] = None
    ) -> ColorPalette:
        """Generate a palette based on emotion"""
        logger.info(f"Generating palette for {emotion.value} emotion")
        return self.generator.generate_palette(
            base_color=base_color,
            emotion=emotion,
            num_colors=num_colors
        )
    
    def generate_harmony_palette(
        self,
        base_color: Color,
        harmony_type: ColorHarmonyType,
        num_colors: int = 5
    ) -> ColorPalette:
        """Generate a harmony-based palette"""
        logger.info(f"Generating {harmony_type.value} harmony palette")
        return self.generator.generate_harmony_palette(
            base_color,
            harmony_type,
            num_colors
        )
    
    def extract_from_image(
        self,
        image_path: str,
        num_colors: int = 5
    ) -> ColorPalette:
        """Extract color palette from an uploaded image"""
        logger.info(f"Extracting colors from {image_path}")
        colors = self.color_extractor.extract_dominant_colors(
            image_path,
            num_colors
        )
        
        return ColorPalette(
            colors=colors,
            name="Extracted Palette",
            description=f"Color palette extracted from {Path(image_path).name}"
        )
    
    def generate_custom_palette(
        self,
        base_color: Optional[Color] = None,
        artist_style: Optional[ArtistStyle] = None,
        emotion: Optional[Emotion] = None,
        harmony_type: Optional[ColorHarmonyType] = None,
        num_colors: int = 5
    ) -> ColorPalette:
        """Generate a custom palette with multiple parameters"""
        logger.info("Generating custom palette")
        return self.generator.generate_palette(
            base_color=base_color,
            artist_style=artist_style,
            emotion=emotion,
            harmony_type=harmony_type,
            num_colors=num_colors
        )
    
    def get_available_styles(self) -> List[ArtistStyle]:
        """Get list of available artist styles"""
        return list(ArtistStyle)
    
    def get_available_emotions(self) -> List[Emotion]:
        """Get list of available emotions"""
        return list(Emotion)
    
    def get_available_harmony_types(self) -> List[ColorHarmonyType]:
        """Get list of available harmony types"""
        return list(ColorHarmonyType)

