"""
Demo script for the AI Color Harmony Generator
"""

from color_harmony_generator import ColorHarmonyManager
from models import ArtistStyle, Emotion, ColorHarmonyType, Color


def demo_artist_styles():
    """Demo: Generate palettes from different artist styles"""
    print("=" * 60)
    print("Demo: Artist Style Palettes")
    print("=" * 60)
    
    manager = ColorHarmonyManager()
    
    for style in [ArtistStyle.MONET, ArtistStyle.ROTHKO, ArtistStyle.INDIAN_MINIATURE]:
        print(f"\n{style.value.replace('_', ' ').title()} Palette:")
        palette = manager.generate_from_artist_style(style, num_colors=5)
        
        for i, color in enumerate(palette.colors, 1):
            print(f"  {i}. {color.to_hex()} - RGB{color.to_rgb_tuple()}")


def demo_emotions():
    """Demo: Generate palettes from emotions"""
    print("\n" + "=" * 60)
    print("Demo: Emotion-Based Palettes")
    print("=" * 60)
    
    manager = ColorHarmonyManager()
    
    for emotion in [Emotion.CALM, Emotion.JOYFUL, Emotion.ENERGETIC]:
        print(f"\n{emotion.value.title()} Emotion Palette:")
        palette = manager.generate_from_emotion(emotion, num_colors=5)
        
        for i, color in enumerate(palette.colors, 1):
            print(f"  {i}. {color.to_hex()} - RGB{color.to_rgb_tuple()}")


def demo_harmony():
    """Demo: Generate harmony-based palettes"""
    print("\n" + "=" * 60)
    print("Demo: Color Harmony Palettes")
    print("=" * 60)
    
    manager = ColorHarmonyManager()
    base_color = Color(100, 150, 200)  # Blue
    
    print(f"\nBase Color: {base_color.to_hex()} - RGB{base_color.to_rgb_tuple()}")
    
    for harmony_type in [
        ColorHarmonyType.ANALOGOUS,
        ColorHarmonyType.COMPLEMENTARY,
        ColorHarmonyType.TRIADIC
    ]:
        print(f"\n{harmony_type.value.replace('_', ' ').title()} Harmony:")
        palette = manager.generate_harmony_palette(
            base_color, harmony_type, num_colors=5
        )
        
        for i, color in enumerate(palette.colors, 1):
            print(f"  {i}. {color.to_hex()} - RGB{color.to_rgb_tuple()}")


def demo_custom():
    """Demo: Generate custom palette"""
    print("\n" + "=" * 60)
    print("Demo: Custom Palette")
    print("=" * 60)
    
    manager = ColorHarmonyManager()
    
    print("\nCustom Palette (Monet + Calm + Analogous):")
    palette = manager.generate_custom_palette(
        artist_style=ArtistStyle.MONET,
        emotion=Emotion.CALM,
        harmony_type=ColorHarmonyType.ANALOGOUS,
        num_colors=6
    )
    
    print(f"Palette Name: {palette.name}")
    print(f"Description: {palette.description}")
    
    for i, color in enumerate(palette.colors, 1):
        print(f"  {i}. {color.to_hex()} - RGB{color.to_rgb_tuple()}")


def main():
    """Run all demos"""
    print("\n🎨 AI Color Harmony Generator - Demo\n")
    
    try:
        demo_artist_styles()
        demo_emotions()
        demo_harmony()
        demo_custom()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

