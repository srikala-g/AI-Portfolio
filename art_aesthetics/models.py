"""
Data models and base classes for the AI Color Harmony Generator
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np


class ArtistStyle(Enum):
    """Enumeration of famous artist styles"""
    MONET = "monet"
    ROTHKO = "rothko"
    INDIAN_MINIATURE = "indian_miniature"
    VAN_GOGH = "van_gogh"
    PICASSO = "picasso"
    KANDINSKY = "kandinsky"
    MATISSE = "matisse"
    WARHOL = "warhol"


class Emotion(Enum):
    """Enumeration of color emotions/themes"""
    CALM = "calm"
    ENERGETIC = "energetic"
    MELANCHOLY = "melancholy"
    JOYFUL = "joyful"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    PROFESSIONAL = "professional"
    NATURAL = "natural"


class ColorHarmonyType(Enum):
    """Enumeration of color harmony types"""
    ANALOGOUS = "analogous"
    COMPLEMENTARY = "complementary"
    TRIADIC = "triadic"
    MONOCHROMATIC = "monochromatic"
    SPLIT_COMPLEMENTARY = "split_complementary"


@dataclass
class Color:
    """Represents a single color with RGB values"""
    r: int
    g: int
    b: int
    
    def __post_init__(self):
        """Validate color values"""
        for value in [self.r, self.g, self.b]:
            if not 0 <= value <= 255:
                raise ValueError(f"Color values must be between 0 and 255, got {value}")
    
    def to_hex(self) -> str:
        """Convert RGB to hex color code"""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    def to_rgb_tuple(self) -> Tuple[int, int, int]:
        """Convert to RGB tuple"""
        return (self.r, self.g, self.b)
    
    def to_hsv(self) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        r_norm = self.r / 255.0
        g_norm = self.g / 255.0
        b_norm = self.b / 255.0
        
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        delta = max_val - min_val
        
        # Calculate Hue
        if delta == 0:
            h = 0
        elif max_val == r_norm:
            h = 60 * (((g_norm - b_norm) / delta) % 6)
        elif max_val == g_norm:
            h = 60 * (((b_norm - r_norm) / delta) + 2)
        else:
            h = 60 * (((r_norm - g_norm) / delta) + 4)
        
        # Calculate Saturation
        s = 0 if max_val == 0 else delta / max_val
        
        # Calculate Value
        v = max_val
        
        return (h, s, v)
    
    @classmethod
    def from_hex(cls, hex_code: str) -> "Color":
        """Create Color from hex code"""
        hex_code = hex_code.lstrip("#")
        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        return cls(r=r, g=g, b=b)


@dataclass
class ColorPalette:
    """Represents a color palette with metadata"""
    colors: List[Color]
    name: str
    artist_style: Optional[ArtistStyle] = None
    emotion: Optional[Emotion] = None
    harmony_type: Optional[ColorHarmonyType] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate palette data"""
        if not self.colors:
            raise ValueError("Color palette must contain at least one color")
        if len(self.colors) > 10:
            raise ValueError("Color palette cannot contain more than 10 colors")
    
    def get_hex_colors(self) -> List[str]:
        """Get list of hex color codes"""
        return [color.to_hex() for color in self.colors]
    
    def get_rgb_colors(self) -> List[Tuple[int, int, int]]:
        """Get list of RGB tuples"""
        return [color.to_rgb_tuple() for color in self.colors]
    
    def get_dominant_color(self) -> Color:
        """Get the first color as dominant (can be enhanced with frequency analysis)"""
        return self.colors[0]
    
    def add_color(self, color: Color) -> None:
        """Add a color to the palette"""
        if len(self.colors) >= 10:
            raise ValueError("Cannot add more than 10 colors to palette")
        self.colors.append(color)
    
    def get_palette_size(self) -> int:
        """Get the number of colors in the palette"""
        return len(self.colors)


class ColorExtractor(ABC):
    """Abstract base class for extracting colors from images"""
    
    @abstractmethod
    def extract_dominant_colors(
        self, 
        image_path: str, 
        num_colors: int = 5
    ) -> List[Color]:
        """Extract dominant colors from an image"""
        pass
    
    @abstractmethod
    def extract_colors_by_style(
        self,
        artist_style: ArtistStyle,
        num_colors: int = 5
    ) -> List[Color]:
        """Extract colors based on artist style"""
        pass


class EmbeddingService(ABC):
    """Abstract base class for generating embeddings"""
    
    @abstractmethod
    def get_color_embedding(self, color: Color) -> np.ndarray:
        """Get embedding for a single color"""
        pass
    
    @abstractmethod
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text description"""
        pass
    
    @abstractmethod
    def get_emotion_embedding(self, emotion: Emotion) -> np.ndarray:
        """Get embedding for an emotion"""
        pass
    
    @abstractmethod
    def get_style_embedding(self, artist_style: ArtistStyle) -> np.ndarray:
        """Get embedding for an artist style"""
        pass
    
    @abstractmethod
    def cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        pass


class PaletteGenerator(ABC):
    """Abstract base class for generating color palettes"""
    
    @abstractmethod
    def generate_palette(
        self,
        base_color: Optional[Color] = None,
        artist_style: Optional[ArtistStyle] = None,
        emotion: Optional[Emotion] = None,
        num_colors: int = 5,
        harmony_type: Optional[ColorHarmonyType] = None
    ) -> ColorPalette:
        """Generate a color palette based on given parameters"""
        pass
    
    @abstractmethod
    def generate_harmony_palette(
        self,
        base_color: Color,
        harmony_type: ColorHarmonyType,
        num_colors: int = 5
    ) -> ColorPalette:
        """Generate a harmony-based color palette"""
        pass

