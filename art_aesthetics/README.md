# 🎨 AI Color Harmony Generator

An intelligent color palette generator that combines data-driven aesthetics with user emotion and theme matching. Generate beautiful color palettes inspired by famous painters (Monet, Rothko, Van Gogh, Indian miniature art, and more) or create harmony-based palettes using color theory.

## ✨ Features

### 🎭 **Artist Style Inspiration**
- Generate palettes inspired by famous painters:
  - **Monet**: Soft impressionist pastels
  - **Rothko**: Bold abstract expressionist blocks
  - **Indian Miniature**: Vibrant traditional colors
  - **Van Gogh**: Post-impressionist vibrant tones
  - **Picasso**: Cubist bold primaries
  - **Kandinsky**: Abstract geometric colors
  - **Matisse**: Fauvist decorative patterns
  - **Warhol**: Pop art high contrast

### 💭 **Emotion-Based Generation**
- Match colors to emotions and themes:
  - Calm, Energetic, Melancholy, Joyful
  - Mysterious, Romantic, Professional, Natural
- Uses CLIP embeddings for semantic color-emotion matching

### 🌈 **Color Harmony Theory**
- Generate palettes based on color theory:
  - **Analogous**: Adjacent colors on color wheel
  - **Complementary**: Opposite colors
  - **Triadic**: Three evenly spaced colors
  - **Monochromatic**: Variations of same hue
  - **Split Complementary**: Base + two adjacent to complement

### 🖼️ **Image Color Extraction**
- Upload images to extract dominant colors
- Uses OpenCV and K-means clustering for accurate color extraction

### 🎯 **Custom Palette Generation**
- Combine multiple parameters:
  - Base color + Artist style + Emotion + Harmony type
- Full control over palette generation

## 🏗️ Architecture

The project follows a modular, object-oriented design:

```
art_aesthetics/
├── models.py                    # Data models and abstract base classes
├── services.py                  # Concrete implementations (OpenCV, CLIP)
├── color_harmony_generator.py   # Main orchestrator class
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### **Core Components**

#### **1. Models (`models.py`)**
- **Data Models**: `Color`, `ColorPalette`, `ArtistStyle`, `Emotion`, `ColorHarmonyType`
- **Abstract Base Classes**: `ColorExtractor`, `EmbeddingService`, `PaletteGenerator`

#### **2. Services (`services.py`)**
- **OpenCVColorExtractor**: Extracts colors from images using K-means clustering
- **CLIPEmbeddingService**: Generates embeddings for color-emotion matching
- **ColorHarmonyGenerator**: Implements color theory harmony algorithms

#### **3. Manager (`color_harmony_generator.py`)**
- **ColorHarmonyManager**: High-level API for palette generation

#### **4. Application (`app.py`)**
- Streamlit web interface with multiple generation modes
- Interactive color palette visualization
- Export functionality

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd art_aesthetics
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install CLIP (optional but recommended for emotion matching):**
```bash
pip install git+https://github.com/openai/CLIP.git
```

**Note:** The app will work without CLIP, but emotion-based color matching will use fallback methods. CLIP provides better semantic understanding for color-emotion relationships.

4. **Run the Streamlit app:**
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Quick Test

Run the demo script to test the installation:
```bash
python demo.py
```

## 📖 Usage

### **Web Interface**

1. **Start the app:**
```bash
streamlit run app.py
```

2. **Choose a generation mode:**
   - **Artist Style**: Select from famous painter styles
   - **Emotion-Based**: Match colors to emotions
   - **Color Harmony**: Generate based on color theory
   - **Image Upload**: Extract colors from images
   - **Custom**: Combine multiple parameters

3. **Configure options:**
   - Number of colors (3-10)
   - Optional base color, emotion, or harmony type

4. **Generate and view:**
   - See hex codes and RGB values
   - Export palette information

### **Python API**

```python
from color_harmony_generator import ColorHarmonyManager
from models import ArtistStyle, Emotion, ColorHarmonyType, Color

# Initialize manager
manager = ColorHarmonyManager()

# Generate from artist style
palette = manager.generate_from_artist_style(
    artist_style=ArtistStyle.MONET,
    num_colors=5,
    emotion=Emotion.CALM
)

# Generate from emotion
palette = manager.generate_from_emotion(
    emotion=Emotion.JOYFUL,
    num_colors=5
)

# Generate color harmony
base_color = Color(100, 150, 200)
palette = manager.generate_harmony_palette(
    base_color=base_color,
    harmony_type=ColorHarmonyType.ANALOGOUS,
    num_colors=5
)

# Extract from image
palette = manager.extract_from_image(
    image_path="path/to/image.jpg",
    num_colors=5
)

# Access palette colors
hex_colors = palette.get_hex_colors()
rgb_colors = palette.get_rgb_colors()
```

## 🔧 Technical Details

### **Color Extraction**
- Uses OpenCV for image processing
- K-means clustering for dominant color extraction
- Pre-defined artist palettes based on historical color analysis

### **CLIP Embeddings**
- Uses OpenAI's CLIP model for semantic understanding
- Generates embeddings for colors, emotions, and styles
- Cosine similarity for color-emotion matching
- Falls back gracefully if CLIP is unavailable

### **Color Harmony Algorithms**
- Implements standard color theory principles
- HSV color space for accurate hue calculations
- Supports all major harmony types

## 🎨 Example Outputs

### **Monet-Inspired Palette**
```
#8AADC2 - RGB(138, 173, 194)  # Soft blue-gray
#FFDAB9 - RGB(255, 218, 185)  # Warm cream
#8BBCCC - RGB(139, 188, 204)  # Sky blue
#FFE4C4 - RGB(255, 228, 196)  # Peach
#B0C4DE - RGB(176, 196, 222)  # Light steel blue
```

### **Joyful Emotion Palette**
```
#FFD700 - RGB(255, 215, 0)    # Gold
#FF69B4 - RGB(255, 105, 180)  # Hot pink
#FFA500 - RGB(255, 165, 0)    # Orange
#FFFF00 - RGB(255, 255, 0)    # Yellow
#00FF7F - RGB(0, 255, 127)    # Spring green
```

## 🛠️ Development

### **Code Structure**
- Follows SOLID principles
- Modular and extensible design
- Abstract base classes for easy extension
- Type hints throughout

### **Adding New Artist Styles**
1. Add style to `ArtistStyle` enum in `models.py`
2. Add color palette to `ARTIST_PALETTES` in `services.py`
3. Add style description to `CLIPEmbeddingService._precompute_style_embeddings()`

### **Adding New Emotions**
1. Add emotion to `Emotion` enum in `models.py`
2. Add description to `CLIPEmbeddingService._precompute_emotion_embeddings()`

## 📝 License

This project is part of the AI Portfolio collection.

## 🙏 Acknowledgments

- **OpenCV**: Image processing and color extraction
- **CLIP**: Semantic color-emotion matching
- **Streamlit**: Beautiful web interface
- **Color Theory**: Traditional color harmony principles

## 🔮 Future Enhancements

- [ ] Save and load palettes
- [ ] Export to CSS, SCSS, or design tools
- [ ] Color accessibility analysis
- [ ] Palette recommendations based on usage
- [ ] Integration with design tools (Figma, Adobe)
- [ ] Machine learning models trained on art datasets

---

**Made with ❤️ using Python, OpenCV, CLIP, and Streamlit**

