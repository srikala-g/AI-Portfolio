"""
Concrete service implementations for the AI Color Harmony Generator
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import KMeans
import logging
from PIL import Image
import torch

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")

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

logger = logging.getLogger(__name__)


class OpenCVColorExtractor(ColorExtractor):
    """OpenCV-based color extraction implementation"""
    
    # Pre-defined color palettes for famous artists
    ARTIST_PALETTES: Dict[ArtistStyle, List[Tuple[int, int, int]]] = {
        ArtistStyle.MONET: [
            (138, 173, 194),  # Soft blue-gray
            (255, 218, 185),  # Warm cream
            (139, 188, 204),  # Sky blue
            (255, 228, 196),  # Peach
            (176, 196, 222),  # Light steel blue
            (255, 250, 240),  # Floral white
            (240, 230, 140),  # Khaki
            (147, 197, 114),  # Sage green
        ],
        ArtistStyle.ROTHKO: [
            (139, 0, 0),      # Dark red
            (25, 25, 112),    # Midnight blue
            (0, 0, 0),        # Black
            (255, 255, 255),  # White
            (128, 0, 128),    # Purple
            (184, 134, 11),   # Dark goldenrod
            (72, 61, 139),    # Dark slate blue
            (220, 20, 60),    # Crimson
        ],
        ArtistStyle.INDIAN_MINIATURE: [
            (255, 215, 0),    # Gold
            (139, 0, 139),    # Dark magenta
            (0, 100, 0),      # Dark green
            (255, 20, 147),   # Deep pink
            (255, 140, 0),    # Dark orange
            (75, 0, 130),     # Indigo
            (255, 192, 203),  # Pink
            (255, 165, 0),    # Orange
        ],
        ArtistStyle.VAN_GOGH: [
            (255, 215, 0),    # Gold
            (255, 140, 0),    # Dark orange
            (72, 61, 139),    # Dark slate blue
            (139, 69, 19),    # Saddle brown
            (255, 255, 0),    # Yellow
            (0, 128, 128),    # Teal
            (255, 69, 0),     # Red orange
            (34, 139, 34),    # Forest green
        ],
        ArtistStyle.PICASSO: [
            (255, 0, 0),      # Red
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 255, 255),  # White
            (0, 0, 0),        # Black
            (128, 128, 128),  # Gray
            (255, 165, 0),    # Orange
            (0, 255, 0),      # Green
        ],
        ArtistStyle.KANDINSKY: [
            (255, 0, 0),      # Red
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (0, 255, 0),      # Green
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (255, 192, 203),  # Pink
            (0, 255, 255),    # Cyan
        ],
        ArtistStyle.MATISSE: [
            (255, 20, 147),   # Deep pink
            (0, 255, 127),    # Spring green
            (255, 215, 0),    # Gold
            (255, 0, 0),      # Red
            (0, 191, 255),    # Deep sky blue
            (255, 140, 0),    # Dark orange
            (255, 192, 203),  # Pink
            (50, 205, 50),    # Lime green
        ],
        ArtistStyle.WARHOL: [
            (255, 0, 0),      # Red
            (255, 255, 0),    # Yellow
            (0, 0, 255),      # Blue
            (255, 255, 255),  # White
            (255, 20, 147),   # Deep pink
            (0, 255, 0),      # Green
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
        ],
    }
    
    def extract_dominant_colors(
        self, 
        image_path: str, 
        num_colors: int = 5
    ) -> List[Color]:
        """Extract dominant colors from an image using K-means clustering"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Validate pixel data
            if len(pixels) == 0:
                raise ValueError("Image contains no pixels")
            
            # Remove any invalid pixels (NaN, Inf, or out of range values)
            valid_mask = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1) & (pixels <= 255).all(axis=1)
            pixels = pixels[valid_mask]
            
            if len(pixels) == 0:
                raise ValueError("No valid pixels found in image")
            
            # Ensure we have enough pixels for clustering
            if len(pixels) < num_colors:
                num_colors = max(1, len(pixels))
                logger.warning(f"Reducing number of colors to {num_colors} due to insufficient pixels")
            
            # Sample pixels if there are too many (for performance and to avoid overflow)
            max_pixels = 50000
            if len(pixels) > max_pixels:
                indices = np.random.choice(len(pixels), max_pixels, replace=False)
                pixels = pixels[indices]
            
            # Ensure pixels are float32 to avoid overflow issues
            pixels = pixels.astype(np.float32)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_
            
            # Validate cluster centers
            colors = np.clip(colors, 0, 255)
            
            # Sort by frequency (approximate)
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            
            # Convert to Color objects
            color_list = []
            for idx in sorted_indices:
                r, g, b = colors[idx].astype(int)
                # Ensure values are in valid range
                r = max(0, min(255, int(r)))
                g = max(0, min(255, int(g)))
                b = max(0, min(255, int(b)))
                color_list.append(Color(r=r, g=g, b=b))
            
            return color_list
            
        except Exception as e:
            logger.error(f"Error extracting colors from image: {e}")
            raise
    
    def extract_colors_by_style(
        self,
        artist_style: ArtistStyle,
        num_colors: int = 5
    ) -> List[Color]:
        """Extract colors based on artist style"""
        if artist_style not in self.ARTIST_PALETTES:
            raise ValueError(f"Artist style {artist_style} not supported")
        
        palette_rgb = self.ARTIST_PALETTES[artist_style]
        # Limit to requested number of colors
        selected_colors = palette_rgb[:num_colors]
        
        return [Color(r=r, g=g, b=b) for r, g, b in selected_colors]


class CLIPEmbeddingService(EmbeddingService):
    """CLIP-based embedding service for color emotion matching"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """Initialize CLIP model"""
        logger = logging.getLogger(__name__)
        
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available. Using fallback embeddings.")
            self.model = None
            self.device = "cpu"
            self.preprocess = None
            self._emotion_embeddings = {}
            self._style_embeddings = {}
            return
        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
            logger.info(f"CLIP model loaded on {self.device}")
            
            # Pre-compute emotion and style embeddings
            self._emotion_embeddings = self._precompute_emotion_embeddings()
            self._style_embeddings = self._precompute_style_embeddings()
            
        except Exception as e:
            logger.warning(f"Could not load CLIP model: {e}. Using fallback.")
            self.model = None
            self.device = "cpu"
            self.preprocess = None
            self._emotion_embeddings = {}
            self._style_embeddings = {}
    
    def _precompute_emotion_embeddings(self) -> Dict[Emotion, np.ndarray]:
        """Pre-compute embeddings for emotions"""
        if self.model is None or not CLIP_AVAILABLE:
            return {}
        
        emotion_descriptions = {
            Emotion.CALM: "calm peaceful serene tranquil",
            Emotion.ENERGETIC: "energetic vibrant dynamic exciting",
            Emotion.MELANCHOLY: "melancholy sad somber blue",
            Emotion.JOYFUL: "joyful happy cheerful bright",
            Emotion.MYSTERIOUS: "mysterious dark mysterious shadowy",
            Emotion.ROMANTIC: "romantic soft pink warm",
            Emotion.PROFESSIONAL: "professional business blue gray",
            Emotion.NATURAL: "natural green earth organic",
        }
        
        embeddings = {}
        with torch.no_grad():
            for emotion, description in emotion_descriptions.items():
                text = clip.tokenize([description]).to(self.device)
                embedding = self.model.encode_text(text)
                embeddings[emotion] = embedding.cpu().numpy()[0]
        
        return embeddings
    
    def _precompute_style_embeddings(self) -> Dict[ArtistStyle, np.ndarray]:
        """Pre-compute embeddings for artist styles"""
        if self.model is None or not CLIP_AVAILABLE:
            return {}
        
        style_descriptions = {
            ArtistStyle.MONET: "impressionist painting soft pastel colors water lilies",
            ArtistStyle.ROTHKO: "abstract expressionist bold color blocks minimal",
            ArtistStyle.INDIAN_MINIATURE: "indian miniature painting vibrant gold detailed",
            ArtistStyle.VAN_GOGH: "post impressionist bold brushstrokes vibrant colors",
            ArtistStyle.PICASSO: "cubist geometric shapes bold primary colors",
            ArtistStyle.KANDINSKY: "abstract geometric shapes colorful vibrant",
            ArtistStyle.MATISSE: "fauvist bold colors decorative patterns",
            ArtistStyle.WARHOL: "pop art vibrant colors high contrast",
        }
        
        embeddings = {}
        with torch.no_grad():
            for style, description in style_descriptions.items():
                text = clip.tokenize([description]).to(self.device)
                embedding = self.model.encode_text(text)
                embeddings[style] = embedding.cpu().numpy()[0]
        
        return embeddings
    
    def get_color_embedding(self, color: Color) -> np.ndarray:
        """Get embedding for a single color by creating a color image"""
        logger = logging.getLogger(__name__)
        
        if self.model is None or not CLIP_AVAILABLE:
            # Fallback: return a simple feature vector
            return np.array([color.r, color.g, color.b, color.r/255, color.g/255, color.b/255])
        
        try:
            # Create a solid color image
            color_image = Image.new('RGB', (224, 224), color.to_rgb_tuple())
            
            # Preprocess and encode
            image_tensor = self.preprocess(color_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
            
            return embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error generating color embedding: {e}")
            # Fallback
            return np.array([color.r, color.g, color.b]) / 255.0
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text description"""
        logger = logging.getLogger(__name__)
        
        if self.model is None or not CLIP_AVAILABLE:
            return np.zeros(512)  # Fallback
        
        try:
            with torch.no_grad():
                text_tokens = clip.tokenize([text]).to(self.device)
                embedding = self.model.encode_text(text_tokens)
            return embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return np.zeros(512)
    
    def get_emotion_embedding(self, emotion: Emotion) -> np.ndarray:
        """Get embedding for an emotion"""
        if emotion in self._emotion_embeddings:
            return self._emotion_embeddings[emotion]
        return self.get_text_embedding(emotion.value)
    
    def get_style_embedding(self, artist_style: ArtistStyle) -> np.ndarray:
        """Get embedding for an artist style"""
        if artist_style in self._style_embeddings:
            return self._style_embeddings[artist_style]
        return self.get_text_embedding(artist_style.value)
    
    def cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        dot_product = np.dot(embedding1, embedding2)
        return dot_product / (norm1 * norm2)


class ColorHarmonyGenerator(PaletteGenerator):
    """Main color palette generator with harmony and style matching"""
    
    def __init__(
        self,
        color_extractor: ColorExtractor,
        embedding_service: EmbeddingService
    ):
        """Initialize the generator with services"""
        self.color_extractor = color_extractor
        self.embedding_service = embedding_service
    
    def generate_palette(
        self,
        base_color: Optional[Color] = None,
        artist_style: Optional[ArtistStyle] = None,
        emotion: Optional[Emotion] = None,
        num_colors: int = 5,
        harmony_type: Optional[ColorHarmonyType] = None
    ) -> ColorPalette:
        """Generate a color palette based on given parameters"""
        colors = []
        
        # Start with base color or artist style
        if artist_style:
            style_colors = self.color_extractor.extract_colors_by_style(
                artist_style, num_colors
            )
            colors = style_colors[:num_colors]
            
            # If base color provided, replace first color
            if base_color:
                colors[0] = base_color
        elif base_color:
            # Generate harmony palette from base color
            if harmony_type:
                return self.generate_harmony_palette(base_color, harmony_type, num_colors)
            else:
                # Default to analogous
                return self.generate_harmony_palette(
                    base_color, ColorHarmonyType.ANALOGOUS, num_colors
                )
        else:
            # Generate default palette
            colors = [Color(100, 150, 200)]  # Default blue
            colors.extend(self._generate_harmony_colors(
                colors[0], ColorHarmonyType.ANALOGOUS, num_colors - 1
            ))
        
        # Refine colors based on emotion if provided
        if emotion and len(colors) > 0:
            colors = self._match_emotion(colors, emotion)
        
        # Create palette name
        name_parts = []
        if artist_style:
            name_parts.append(artist_style.value.replace("_", " ").title())
        if emotion:
            name_parts.append(emotion.value.title())
        if harmony_type:
            name_parts.append(harmony_type.value.replace("_", " ").title())
        name = " ".join(name_parts) if name_parts else "Custom Palette"
        
        description = f"A {name.lower()} color palette"
        if artist_style:
            description += f" inspired by {artist_style.value.replace('_', ' ')}"
        if emotion:
            description += f" evoking {emotion.value}"
        
        return ColorPalette(
            colors=colors[:num_colors],
            name=name,
            artist_style=artist_style,
            emotion=emotion,
            harmony_type=harmony_type,
            description=description
        )
    
    def generate_harmony_palette(
        self,
        base_color: Color,
        harmony_type: ColorHarmonyType,
        num_colors: int = 5
    ) -> ColorPalette:
        """Generate a harmony-based color palette"""
        colors = [base_color]
        colors.extend(self._generate_harmony_colors(base_color, harmony_type, num_colors - 1))
        
        harmony_names = {
            ColorHarmonyType.ANALOGOUS: "Analogous",
            ColorHarmonyType.COMPLEMENTARY: "Complementary",
            ColorHarmonyType.TRIADIC: "Triadic",
            ColorHarmonyType.MONOCHROMATIC: "Monochromatic",
            ColorHarmonyType.SPLIT_COMPLEMENTARY: "Split Complementary",
        }
        
        return ColorPalette(
            colors=colors[:num_colors],
            name=f"{harmony_names[harmony_type]} Harmony",
            harmony_type=harmony_type,
            description=f"A {harmony_type.value.replace('_', ' ')} color harmony palette"
        )
    
    def _generate_harmony_colors(
        self,
        base_color: Color,
        harmony_type: ColorHarmonyType,
        num_additional: int
    ) -> List[Color]:
        """Generate harmony colors based on color theory"""
        h, s, v = base_color.to_hsv()
        colors = []
        
        if harmony_type == ColorHarmonyType.ANALOGOUS:
            # Colors adjacent on color wheel (30 degrees apart)
            for i in range(1, num_additional + 1):
                new_h = (h + (i * 30)) % 360
                colors.append(self._hsv_to_color(new_h, s, v))
        
        elif harmony_type == ColorHarmonyType.COMPLEMENTARY:
            # Opposite color on color wheel (180 degrees)
            comp_h = (h + 180) % 360
            colors.append(self._hsv_to_color(comp_h, s, v))
            # Add variations
            for i in range(1, num_additional):
                variation_h = (comp_h + (i * 20)) % 360
                colors.append(self._hsv_to_color(variation_h, s * 0.8, v))
        
        elif harmony_type == ColorHarmonyType.TRIADIC:
            # Three colors evenly spaced (120 degrees apart)
            for i in range(1, num_additional + 1):
                new_h = (h + (i * 120)) % 360
                colors.append(self._hsv_to_color(new_h, s, v))
        
        elif harmony_type == ColorHarmonyType.MONOCHROMATIC:
            # Variations of same hue with different saturation/value
            for i in range(1, num_additional + 1):
                new_s = min(1.0, s * (1.0 - i * 0.2))
                new_v = min(1.0, v * (1.0 - i * 0.15))
                colors.append(self._hsv_to_color(h, new_s, new_v))
        
        elif harmony_type == ColorHarmonyType.SPLIT_COMPLEMENTARY:
            # Base color + two colors adjacent to complement
            comp_h = (h + 180) % 360
            colors.append(self._hsv_to_color((comp_h - 30) % 360, s, v))
            colors.append(self._hsv_to_color((comp_h + 30) % 360, s, v))
            # Fill remaining slots
            for i in range(2, num_additional):
                new_h = (h + (i * 45)) % 360
                colors.append(self._hsv_to_color(new_h, s * 0.9, v))
        
        return colors[:num_additional]
    
    def _hsv_to_color(self, h: float, s: float, v: float) -> Color:
        """Convert HSV to Color object"""
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        # Clamp values
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return Color(r=r, g=g, b=b)
    
    def _match_emotion(self, colors: List[Color], emotion: Emotion) -> List[Color]:
        """Refine colors to better match emotion using embeddings"""
        if not hasattr(self.embedding_service, 'get_emotion_embedding'):
            return colors
        
        try:
            emotion_embedding = self.embedding_service.get_emotion_embedding(emotion)
            
            # Score each color against emotion
            color_scores = []
            for color in colors:
                color_embedding = self.embedding_service.get_color_embedding(color)
                similarity = self.embedding_service.cosine_similarity(
                    color_embedding, emotion_embedding
                )
                color_scores.append((similarity, color))
            
            # Sort by similarity and return top colors
            color_scores.sort(key=lambda x: x[0], reverse=True)
            return [color for _, color in color_scores]
            
        except Exception as e:
            logger.error(f"Error matching emotion: {e}")
            return colors

