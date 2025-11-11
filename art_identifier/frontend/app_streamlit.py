"""
Streamlit frontend for Art Identifier application.
"""

import streamlit as st
import requests
from PIL import Image
import io
import os
from pathlib import Path

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
    except Exception as e:
        return False, 0


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
        # Ensure we return a list of dictionaries
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            return [data] if data else []
        else:
            st.warning(f"Unexpected response format: {type(data)}")
            return []
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Response: {e.response.text}")
        return []
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
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
        # Ensure we return a list of dictionaries
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            return [data] if data else []
        else:
            st.warning(f"Unexpected response format: {type(data)}")
            return []
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Response: {e.response.text}")
        return []
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return []


def search_by_image_and_text(image_file, query: str, top_k: int = 5):
    """Search artworks by image and text."""
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
        # Ensure we return a list of dictionaries
        if isinstance(result_data, list):
            return [item for item in result_data if isinstance(item, dict)]
        elif isinstance(result_data, dict):
            return [result_data] if result_data else []
        else:
            st.warning(f"Unexpected response format: {type(result_data)}")
            return []
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Response: {e.response.text}")
        return []
    except Exception as e:
        st.error(f"Error searching: {str(e)}")
        return []


# Main interface
tab1, tab2, tab3 = st.tabs(["Text Search", "Image Search", "Combined Search"])

# Check API health
is_healthy, artworks_count = check_api_health()
if not is_healthy:
    st.error(f"⚠️ API is not available at {api_url}. Please make sure the backend is running.")
    st.stop()
else:
    st.success(f"✅ API connected. {artworks_count} artworks loaded.")

# Tab 1: Text Search
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
                
                # Display results in columns
                cols = st.columns(min(3, len(results)))
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        # Ensure result is a dictionary
                        if not isinstance(result, dict):
                            st.error(f"Unexpected result format: {type(result)}")
                            continue
                        
                        st.markdown(f"### {result.get('title', 'Unknown')}")
                        st.markdown(f"**Artist:** {result.get('artist', 'Unknown')}")
                        if result.get('genre'):
                            st.markdown(f"**Genre:** {result.get('genre')}")
                        if result.get('style'):
                            st.markdown(f"**Style:** {result.get('style')}")
                        st.markdown(f"**Similarity:** {result.get('similarity_score', 0):.4f}")
                        
                        # Try to display image if path exists
                        image_path = result.get('image_path')
                        if image_path:
                            full_path = Path("../data") / image_path
                            if full_path.exists():
                                try:
                                    img = Image.open(full_path)
                                    st.image(img, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not load image: {e}")
            else:
                st.info("No results found.")
        else:
            st.warning("Please enter a search query.")

# Tab 2: Image Search
with tab2:
    st.header("Search by Image Upload")
    uploaded_file = st.file_uploader(
        "Upload an artwork image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to find similar artworks"
    )
    top_k_image = st.slider("Number of results", 1, 20, 5, key="image_top_k")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Search", key="image_search_btn"):
            with st.spinner("Searching..."):
                # Reset file pointer
                uploaded_file.seek(0)
                results = search_by_image(uploaded_file, top_k_image)
            
            if results:
                st.subheader(f"Found {len(results)} similar artworks")
                
                # Display results in columns
                cols = st.columns(min(3, len(results)))
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        # Ensure result is a dictionary
                        if not isinstance(result, dict):
                            st.error(f"Unexpected result format: {type(result)}")
                            continue
                        
                        st.markdown(f"### {result.get('title', 'Unknown')}")
                        st.markdown(f"**Artist:** {result.get('artist', 'Unknown')}")
                        if result.get('genre'):
                            st.markdown(f"**Genre:** {result.get('genre')}")
                        if result.get('style'):
                            st.markdown(f"**Style:** {result.get('style')}")
                        st.markdown(f"**Similarity:** {result.get('similarity_score', 0):.4f}")
                        
                        # Try to display image if path exists
                        image_path = result.get('image_path')
                        if image_path:
                            full_path = Path("../data") / image_path
                            if full_path.exists():
                                try:
                                    img = Image.open(full_path)
                                    st.image(img, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not load image: {e}")
            else:
                st.info("No results found.")

# Tab 3: Combined Search
with tab3:
    st.header("Search by Image and Text")
    uploaded_file_combined = st.file_uploader(
        "Upload an artwork image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image and add a text description",
        key="combined_image"
    )
    text_query_combined = st.text_input(
        "Additional text description (optional)",
        placeholder="e.g., impressionist style, oil painting",
        key="combined_text"
    )
    top_k_combined = st.slider("Number of results", 1, 20, 5, key="combined_top_k")
    
    if uploaded_file_combined is not None:
        # Display uploaded image
        image = Image.open(uploaded_file_combined)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Search", key="combined_search_btn"):
            with st.spinner("Searching..."):
                # Reset file pointer
                uploaded_file_combined.seek(0)
                results = search_by_image_and_text(
                    uploaded_file_combined,
                    text_query_combined or "",
                    top_k_combined
                )
            
            if results:
                st.subheader(f"Found {len(results)} results")
                
                # Display results in columns
                cols = st.columns(min(3, len(results)))
                for idx, result in enumerate(results):
                    with cols[idx % 3]:
                        # Ensure result is a dictionary
                        if not isinstance(result, dict):
                            st.error(f"Unexpected result format: {type(result)}")
                            continue
                        
                        st.markdown(f"### {result.get('title', 'Unknown')}")
                        st.markdown(f"**Artist:** {result.get('artist', 'Unknown')}")
                        if result.get('genre'):
                            st.markdown(f"**Genre:** {result.get('genre')}")
                        if result.get('style'):
                            st.markdown(f"**Style:** {result.get('style')}")
                        st.markdown(f"**Similarity:** {result.get('similarity_score', 0):.4f}")
                        
                        # Try to display image if path exists
                        image_path = result.get('image_path')
                        if image_path:
                            full_path = Path("../data") / image_path
                            if full_path.exists():
                                try:
                                    img = Image.open(full_path)
                                    st.image(img, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not load image: {e}")
            else:
                st.info("No results found.")

