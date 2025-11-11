"""
Streamlit web application for AI Color Harmony Generator
"""

import streamlit as st
import numpy as np
from PIL import Image
import logging
from typing import Optional

from models import Color, ArtistStyle, Emotion, ColorHarmonyType
from color_harmony_generator import ColorHarmonyManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Color Harmony Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .palette-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .color-box {
        flex: 1;
        min-width: 80px;
        height: 100px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'manager' not in st.session_state:
    st.session_state.manager = ColorHarmonyManager()
if 'current_palette' not in st.session_state:
    st.session_state.current_palette = None


def display_palette(palette):
    """Display color palette with hex codes"""
    colors = palette.get_hex_colors()
    
    st.markdown(f"### {palette.name}")
    if palette.description:
        st.markdown(f"*{palette.description}*")
    
    # Create columns for colors
    num_colors = len(colors)
    cols = st.columns(num_colors)
    
    for i, (col, hex_color) in enumerate(zip(cols, colors)):
        with col:
            # Calculate text color (black or white) based on brightness
            rgb = palette.colors[i].to_rgb_tuple()
            brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
            text_color = "#000000" if brightness > 128 else "#ffffff"
            
            st.markdown(
                f"""
                <div style="
                    background-color: {hex_color};
                    color: {text_color};
                    padding: 40px 20px;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    min-height: 120px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <div>
                        <div style="font-size: 1.2rem; margin-bottom: 10px;">{hex_color.upper()}</div>
                        <div style="font-size: 0.9rem;">RGB({rgb[0]}, {rgb[1]}, {rgb[2]})</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Display metadata
    with st.expander("Palette Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if palette.artist_style:
                st.write(f"**Artist Style:** {palette.artist_style.value.replace('_', ' ').title()}")
        with col2:
            if palette.emotion:
                st.write(f"**Emotion:** {palette.emotion.value.title()}")
        with col3:
            if palette.harmony_type:
                st.write(f"**Harmony:** {palette.harmony_type.value.replace('_', ' ').title()}")
        
        # Export palette as text
        st.markdown("### Export Palette")
        palette_text = "\n".join([f"{hex_color} - {palette.colors[i].to_rgb_tuple()}" 
                                  for i, hex_color in enumerate(colors)])
        st.code(palette_text, language="text")


def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🎨 AI Color Harmony Generator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Generate beautiful color palettes inspired by famous painters or match your emotions
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Generation mode
        mode = st.radio(
            "Generation Mode",
            ["Image Upload", "Artist Style", "Emotion-Based", "Color Harmony", "Custom"],
            help="Choose how you want to generate your palette"
        )
        
        num_colors = st.slider(
            "Number of Colors",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of colors in the palette"
        )
    
    # Main content area
    if mode == "Image Upload":
        st.header("🖼️ Extract from Image")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to extract dominant colors"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Save uploaded file temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name, "JPEG")
                tmp_path = tmp_file.name
            
            if st.button("Extract Colors", type="primary"):
                with st.spinner("Extracting colors from image..."):
                    try:
                        palette = st.session_state.manager.extract_from_image(
                            image_path=tmp_path,
                            num_colors=num_colors
                        )
                        st.session_state.current_palette = palette
                        st.success("Colors extracted successfully!")
                        # Clean up
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"Error extracting colors: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        
        elif mode == "Artist Style":
            st.header("🎭 Generate from Artist Style")
            
            col1, col2 = st.columns(2)
            with col1:
                artist_style = st.selectbox(
                    "Select Artist Style",
                    options=[style.value for style in ArtistStyle],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            
            with col2:
                emotion = st.selectbox(
                    "Optional: Emotion/Theming",
                    options=["None"] + [e.value for e in Emotion],
                    format_func=lambda x: x.replace("_", " ").title() if x != "None" else x
                )
            
            if st.button("Generate Palette", type="primary"):
                with st.spinner("Generating palette..."):
                    try:
                        style = ArtistStyle(artist_style)
                        emo = Emotion(emotion) if emotion != "None" else None
                        palette = st.session_state.manager.generate_from_artist_style(
                            artist_style=style,
                            emotion=emo,
                            num_colors=num_colors
                        )
                        st.session_state.current_palette = palette
                        st.success("Palette generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating palette: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
        
        elif mode == "Emotion-Based":
            st.header("💭 Generate from Emotion")
            
            col1, col2 = st.columns(2)
            with col1:
                emotion = st.selectbox(
                    "Select Emotion/Theme",
                    options=[e.value for e in Emotion],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            
            with col2:
                st.subheader("Optional: Base Color")
                base_color_hex = st.color_picker("Choose a base color", value="#6496C8")
                base_color = Color.from_hex(base_color_hex)
            
            if st.button("Generate Palette", type="primary"):
                with st.spinner("Generating palette..."):
                    try:
                        emo = Emotion(emotion)
                        palette = st.session_state.manager.generate_from_emotion(
                            emotion=emo,
                            base_color=base_color,
                            num_colors=num_colors
                        )
                        st.session_state.current_palette = palette
                        st.success("Palette generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating palette: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
        
        elif mode == "Color Harmony":
            st.header("🌈 Generate Color Harmony")
            
            col1, col2 = st.columns(2)
            with col1:
                base_color_hex = st.color_picker("Choose Base Color", value="#6496C8")
                base_color = Color.from_hex(base_color_hex)
            
            with col2:
                harmony_type = st.selectbox(
                    "Harmony Type",
                    options=[ht.value for ht in ColorHarmonyType],
                    format_func=lambda x: x.replace("_", " ").title()
                )
            
            if st.button("Generate Palette", type="primary"):
                with st.spinner("Generating palette..."):
                    try:
                        harmony = ColorHarmonyType(harmony_type)
                        palette = st.session_state.manager.generate_harmony_palette(
                            base_color=base_color,
                            harmony_type=harmony,
                            num_colors=num_colors
                        )
                        st.session_state.current_palette = palette
                        st.success("Palette generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating palette: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
        
        elif mode == "Custom":
            st.header("🎯 Custom Palette Generation")
            
            col1, col2 = st.columns(2)
            with col1:
                base_color_hex = st.color_picker("Base Color (Optional)", value="#6496C8")
                base_color = Color.from_hex(base_color_hex) if base_color_hex else None
                
                artist_style = st.selectbox(
                    "Artist Style (Optional)",
                    options=["None"] + [s.value for s in ArtistStyle],
                    format_func=lambda x: x.replace("_", " ").title() if x != "None" else x
                )
            
            with col2:
                emotion = st.selectbox(
                    "Emotion (Optional)",
                    options=["None"] + [e.value for e in Emotion],
                    format_func=lambda x: x.replace("_", " ").title() if x != "None" else x
                )
                
                harmony_type = st.selectbox(
                    "Harmony Type (Optional)",
                    options=["None"] + [ht.value for ht in ColorHarmonyType],
                    format_func=lambda x: x.replace("_", " ").title() if x != "None" else x
                )
            
            if st.button("Generate Custom Palette", type="primary"):
                with st.spinner("Generating custom palette..."):
                    try:
                        style = ArtistStyle(artist_style) if artist_style != "None" else None
                        emo = Emotion(emotion) if emotion != "None" else None
                        harmony = ColorHarmonyType(harmony_type) if harmony_type != "None" else None
                        
                        palette = st.session_state.manager.generate_custom_palette(
                            base_color=base_color,
                            artist_style=style,
                            emotion=emo,
                            harmony_type=harmony,
                            num_colors=num_colors
                        )
                        st.session_state.current_palette = palette
                        st.success("Custom palette generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating palette: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
        
        # Display current palette
        if st.session_state.current_palette:
            st.markdown("---")
            display_palette(st.session_state.current_palette)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>AI Color Harmony Generator | Powered by OpenCV, CLIP, and Color Theory</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

