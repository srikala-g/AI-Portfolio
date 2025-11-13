"""
Streamlit frontend for Art Identifier application.
"""

import streamlit as st
import requests
from PIL import Image
from pathlib import Path
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Art Identifier",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Art Identifier")
st.markdown("Identify and search artworks using AI-powered embeddings")

# Sidebar for API configuration
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_BASE_URL)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application uses AI embeddings to:
    - Search artworks by text description
    - Search artworks by image upload
    - Find similar artworks
    """)


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("artworks_loaded", 0)
        return False, 0
    except Exception:
        return False, 0


def display_artwork(result):
    """Helper to display a single artwork card."""
    st.markdown(f"### {result.get('title', 'Unknown')}")
    st.markdown(f"**Artist:** {result.get('artist', 'Unknown')}")
    if result.get('genre'):
        st.markdown(f"**Genre:** {result.get('genre')}")
    if result.get('style'):
        st.markdown(f"**Style:** {result.get('style')}")
    st.markdown(f"**Similarity:** {result.get('similarity_score', 0):.4f}")

    # Show image from API URL
    image_url = result.get('image_url') or result.get('image_path')
    if image_url:
        if not image_url.startswith("http"):
            image_url = f"{api_url}{image_url}"
        try:
            st.image(image_url, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display image: {e}")


def search_by_text(query: str, top_k: int = 5):
    """Search artworks by text."""
    try:
        response = requests.post(
            f"{api_url}/search/text",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else [data]
    except Exception as e:
        st.error(f"Error searching by text: {str(e)}")
        return []


def search_by_image(image_file, top_k: int = 5):
    """Search artworks by image."""
    try:
        files = {"file": image_file}
        response = requests.post(
            f"{api_url}/search/image",
            files=files,
            params={"top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else [data]
    except Exception as e:
        st.error(f"Error searching by image: {str(e)}")
        return []


def search_by_image_and_text(image_file, query: str, top_k: int = 5):
    """Search artworks by both image and text."""
    try:
        files = {"file": image_file}
        data = {"query": query, "top_k": top_k}
        response = requests.post(
            f"{api_url}/search/image-and-text",
            files=files,
            data=data,
            timeout=30
        )
        response.raise_for_status()
        result_data = response.json()
        return result_data if isinstance(result_data, list) else [result_data]
    except Exception as e:
        st.error(f"Error searching by image and text: {str(e)}")
        return []


# -----------------------------
# UI Layout
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Text Search", "Image Search", "Combined Search"])

# Check API health
is_healthy, artworks_count = check_api_health()
if not is_healthy:
    st.error(f"⚠️ API is not available at {api_url}. Please make sure the backend is running.")
    st.stop()
else:
    st.success(f"✅ API connected. {artworks_count} artworks loaded.")


# --- Tab 1: Text Search ---
with tab1:
    st.header("Search by Text Description")
    text_query = st.text_input(
        "Enter a description of the artwork",
        placeholder="e.g., starry night painting, portrait of a woman, abstract art"
    )
    top_k_text = st.slider("Number of results", 1, 20, 5, key="text_top_k")

    if st.button("Search", key="text_search_btn"):
        if text_query:
            with st.spinner("Searching..."):
                results = search_by_text(text_query, top_k_text)
            if results:
                st.subheader(f"Found {len(results)} results")
                cols = st.columns(min(3, len(results)))
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        display_artwork(result)
            else:
                st.info("No results found.")
        else:
            st.warning("Please enter a search query.")


# --- Tab 2: Image Search ---
with tab2:
    st.header("Search by Image Upload")
    uploaded_file = st.file_uploader(
        "Upload an artwork image",
        type=["jpg", "jpeg", "png"]
    )
    top_k_image = st.slider("Number of results", 1, 20, 5, key="image_top_k")

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption="Uploaded Image", use_container_width=True)
        if st.button("Search", key="image_search_btn"):
            with st.spinner("Searching..."):
                uploaded_file.seek(0)
                results = search_by_image(uploaded_file, top_k_image)
            if results:
                st.subheader(f"Found {len(results)} similar artworks")
                cols = st.columns(min(3, len(results)))
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        display_artwork(result)
            else:
                st.info("No results found.")


# --- Tab 3: Combined Search ---
with tab3:
    st.header("Search by Image and Text")
    uploaded_file_combined = st.file_uploader(
        "Upload an artwork image",
        type=["jpg", "jpeg", "png"],
        key="combined_image"
    )
    text_query_combined = st.text_input(
        "Additional text description (optional)",
        placeholder="e.g., impressionist style, oil painting",
        key="combined_text"
    )
    top_k_combined = st.slider("Number of results", 1, 20, 5, key="combined_top_k")

    if uploaded_file_combined is not None:
        st.image(Image.open(uploaded_file_combined), caption="Uploaded Image", use_container_width=True)
        if st.button("Search", key="combined_search_btn"):
            with st.spinner("Searching..."):
                uploaded_file_combined.seek(0)
                results = search_by_image_and_text(uploaded_file_combined, text_query_combined, top_k_combined)
            if results:
                st.subheader(f"Found {len(results)} results")
                cols = st.columns(min(3, len(results)))
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        display_artwork(result)
            else:
                st.info("No results found.")
