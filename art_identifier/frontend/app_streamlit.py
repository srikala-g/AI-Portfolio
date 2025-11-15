"""
Streamlit frontend for Art Identifier application.
"""

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlencode
import streamlit.components.v1 as components
import os
import uuid
import numpy as np
from collections import Counter

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


def resolve_image_url(image_reference: str):
    """Resolve API-relative image paths to absolute URLs."""
    if not image_reference:
        return None
    if not image_reference.startswith("http"):
        return f"{api_url}{image_reference}"
    return image_reference


def extract_color_palette(image_url: str, num_colors: int = 12):
    """Extract comprehensive color palette with emphasis on highlights and saturated colors."""
    try:
        # Download the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Open image
        img = Image.open(BytesIO(response.content))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Keep original size up to 400x400 for better color detection
        max_size = 400
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        # Calculate brightness and saturation for each pixel
        brightness = pixels.max(axis=1)  # Max RGB component
        saturation = (pixels.max(axis=1) - pixels.min(axis=1)) / (pixels.max(axis=1) + 1e-6)
        
        # Stratified sampling to ensure we capture diverse colors
        # Divide into regions based on brightness and saturation
        bright_saturated = (brightness > 180) & (saturation > 0.3)  # Highlights
        bright_unsaturated = (brightness > 180) & (saturation <= 0.3)  # Light tones
        mid_saturated = (brightness >= 80) & (brightness <= 180) & (saturation > 0.2)  # Vibrant mid-tones
        mid_unsaturated = (brightness >= 80) & (brightness <= 180) & (saturation <= 0.2)  # Muted mid-tones
        dark = brightness < 80  # Shadows
        
        # Sample from each region
        samples_per_region = 2000
        sampled_pixels = []
        
        for mask in [bright_saturated, bright_unsaturated, mid_saturated, mid_unsaturated, dark]:
            region_pixels = pixels[mask]
            if len(region_pixels) > 0:
                if len(region_pixels) > samples_per_region:
                    indices = np.random.choice(len(region_pixels), samples_per_region, replace=False)
                    sampled_pixels.append(region_pixels[indices])
                else:
                    sampled_pixels.append(region_pixels)
        
        if sampled_pixels:
            pixels_to_cluster = np.vstack(sampled_pixels)
        else:
            # Fallback to simple sampling
            sample_size = min(10000, len(pixels))
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels_to_cluster = pixels[indices]
        
        # Implement K-means clustering with improved initialization
        centroids = []
        
        # Force include at least one bright/saturated color if present
        if bright_saturated.any():
            bright_sat_pixels = pixels[bright_saturated]
            idx = np.random.randint(0, len(bright_sat_pixels))
            centroids.append(bright_sat_pixels[idx].copy())
        
        # Force include at least one mid-saturated color if present
        if mid_saturated.any() and len(centroids) < num_colors:
            mid_sat_pixels = pixels[mid_saturated]
            idx = np.random.randint(0, len(mid_sat_pixels))
            centroids.append(mid_sat_pixels[idx].copy())
        
        # Fill remaining centroids using k-means++ on the sampled pixels
        remaining = num_colors - len(centroids)
        
        if remaining > 0:
            if len(centroids) == 0:
                # Start with random pixel if no forced centroids
                first_idx = np.random.randint(0, len(pixels_to_cluster))
                centroids.append(pixels_to_cluster[first_idx].copy())
                remaining -= 1
            
            # k-means++ for remaining centroids
            for _ in range(remaining):
                distances = np.array([
                    min(np.linalg.norm(pixel - centroid) for centroid in centroids)
                    for pixel in pixels_to_cluster
                ])
                
                probabilities = distances ** 2
                probabilities /= probabilities.sum()
                
                next_idx = np.random.choice(len(pixels_to_cluster), p=probabilities)
                centroids.append(pixels_to_cluster[next_idx].copy())
        
        centroids = np.array(centroids)
        
        # K-means iterations
        max_iterations = 30
        for iteration in range(max_iterations):
            # Assign each pixel to nearest centroid
            distances = np.array([
                np.linalg.norm(pixels_to_cluster - centroid, axis=1)
                for centroid in centroids
            ])
            assignments = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([
                pixels_to_cluster[assignments == i].mean(axis=0) if np.any(assignments == i) 
                else centroids[i]
                for i in range(num_colors)
            ])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=1.0):
                break
            
            centroids = new_centroids
        
        # Calculate cluster statistics for sorting
        cluster_info = []
        for i in range(num_colors):
            cluster_pixels = pixels_to_cluster[assignments == i]
            if len(cluster_pixels) > 0:
                count = len(cluster_pixels)
                avg_brightness = cluster_pixels.max(axis=1).mean()
                avg_saturation = ((cluster_pixels.max(axis=1) - cluster_pixels.min(axis=1)) / 
                                 (cluster_pixels.max(axis=1) + 1e-6)).mean()
                # Prioritize: frequency, then brightness, then saturation
                score = count * 1.0 + avg_brightness * 0.5 + avg_saturation * 50
                cluster_info.append((i, count, score))
            else:
                cluster_info.append((i, 0, 0))
        
        # Sort by score (frequency + brightness + saturation)
        cluster_info.sort(key=lambda x: x[2], reverse=True)
        
        # Extract final palette
        palette = []
        seen_colors = set()
        
        for cluster_idx, _, _ in cluster_info:
            r, g, b = centroids[cluster_idx]
            r, g, b = int(round(r)), int(round(g)), int(round(b))
            
            # Clamp values to valid range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            
            if hex_color not in seen_colors:
                palette.append(hex_color)
                seen_colors.add(hex_color)
        
        return palette
    except Exception as e:
        st.error(f"Error extracting palette: {str(e)}")
        return None


def display_color_palette(colors: list):
    """Display color palette as swatches."""
    if not colors:
        return
    
    st.markdown("#### Color Palette")
    cols = st.columns(len(colors))
    
    for idx, color in enumerate(colors):
        with cols[idx]:
            # Create a small colored square
            st.markdown(
                f'<div style="width:100%; height:60px; background-color:{color}; border-radius:8px; border:2px solid #ddd;"></div>',
                unsafe_allow_html=True
            )
            st.caption(color.upper())


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
    image_url = resolve_image_url(result.get('image_url') or result.get('image_path'))
    if image_url:
        try:
            st.image(image_url, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display image: {e}")


def display_search_result(result, idx: int, search_id: str | None):
    """Display search result card with clickable image to update details tab."""
    st.markdown(f"#### {result.get('title', 'Unknown')}")
    st.caption(result.get('artist', 'Unknown'))

    image_url = resolve_image_url(result.get('image_url') or result.get('image_path'))
    if image_url:
        params = {"selected_idx": idx, "active_tab": "details"}
        if search_id:
            params["search_id"] = search_id
        query_string = urlencode(params)
        st.markdown(
            f'<a href="?{query_string}#artwork-details" target="_self">'
            f'<img src="{image_url}" style="width:100%; border-radius:12px;"/>'
            f"</a>",
            unsafe_allow_html=True,
        )
        st.caption("Click artwork to view details")
    else:
        st.info("Image preview unavailable.")

    if result.get('similarity_score') is not None:
        st.markdown(f"**Similarity:** {result.get('similarity_score', 0):.4f}")

    if result.get('description'):
        st.write(result.get('description'))


def clear_search_results():
    """Clear current search results and related session/query state."""
    previous_search_id = st.session_state.get("current_search_id")
    st.session_state["search_results"] = []
    st.session_state["details_select_idx"] = 0
    st.session_state["current_search_id"] = None
    st.session_state["last_search_feedback"] = None
    st.session_state["color_palettes"] = {}
    remove_search_state(previous_search_id)
    for key in ("search_id", "selected_idx"):
        if key in st.query_params:
            del st.query_params[key]


def get_input_signature(uploaded_file, text_value: str):
    """Create a comparable signature for the current search inputs."""
    file_name = getattr(uploaded_file, "name", None)
    file_size = getattr(uploaded_file, "size", None)
    normalized_text = (text_value or "").strip()
    return (file_name, file_size, normalized_text)


@st.cache_resource
def get_search_cache():
    """Global cache to persist search results across reruns and sessions."""
    return {}


def persist_search_state():
    """Persist current search-related session state into the shared cache."""
    search_id = st.session_state.get("current_search_id")
    if not search_id:
        return

    cache = get_search_cache()
    cache[search_id] = {
        "results": st.session_state.get("search_results", []),
        "image": st.session_state.get("last_uploaded_image"),
        "text_query": st.session_state.get("last_text_query", ""),
        "selected_idx": st.session_state.get("details_select_idx", 0),
    }


def remove_search_state(search_id: str | None):
    """Remove search state records from cache."""
    if not search_id:
        return
    cache = get_search_cache()
    cache.pop(search_id, None)


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
# Session state & routing setup
# -----------------------------
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []

if "current_search_id" not in st.session_state:
    st.session_state["current_search_id"] = None

if "last_uploaded_image" not in st.session_state:
    st.session_state["last_uploaded_image"] = None

if "last_text_query" not in st.session_state:
    st.session_state["last_text_query"] = ""

if "details_select_idx" not in st.session_state:
    st.session_state["details_select_idx"] = 0

if "last_search_feedback" not in st.session_state:
    st.session_state["last_search_feedback"] = None

if "color_palettes" not in st.session_state:
    st.session_state["color_palettes"] = {}  # Store palettes by search_id and idx

if "active_tab_target" not in st.session_state:
    st.session_state["active_tab_target"] = None

if "last_input_signature" not in st.session_state:
    st.session_state["last_input_signature"] = None

if "widget_reset_counter" not in st.session_state:
    st.session_state["widget_reset_counter"] = 0

incoming_query_params = st.query_params


def _first_query_value(value):
    if isinstance(value, list):
        return value[0]
    return value


search_id_from_query = _first_query_value(incoming_query_params.get("search_id"))
if search_id_from_query:
    cached_state = get_search_cache().get(search_id_from_query)
    if cached_state:
        st.session_state["current_search_id"] = search_id_from_query
        st.session_state["search_results"] = cached_state.get("results", [])
        st.session_state["last_uploaded_image"] = cached_state.get("image")
        st.session_state["last_text_query"] = cached_state.get("text_query", "")
        st.session_state["details_select_idx"] = cached_state.get("selected_idx", 0)


selected_idx_param = _first_query_value(incoming_query_params.get("selected_idx"))
if selected_idx_param is not None:
    try:
        st.session_state["details_select_idx"] = int(selected_idx_param)
        persist_search_state()
    except (TypeError, ValueError):
        pass

def _focus_details_tab():
    components.html(
        """
        <script>
        const switchToDetails = () => {
            const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
            if (!tabs || !tabs.length) {
                setTimeout(switchToDetails, 100);
                return;
            }
            tabs.forEach((tab) => {
                const label = tab.innerText.trim().toLowerCase();
                if (label === 'artwork details') {
                    tab.click();
                }
            });
        };
        setTimeout(switchToDetails, 0);
        </script>
        """,
        height=0,
    )


active_tab_param = _first_query_value(incoming_query_params.get("active_tab"))
if active_tab_param == "details":
    _focus_details_tab()
    if "active_tab" in st.query_params:
        del st.query_params["active_tab"]

if st.session_state.get("active_tab_target") == "details":
    _focus_details_tab()
    st.session_state["active_tab_target"] = None


# Check API health
is_healthy, artworks_count = check_api_health()
if not is_healthy:
    st.error(f"⚠️ API is not available at {api_url}. Please make sure the backend is running.")
    st.stop()
search_tab, details_tab = st.tabs(["Search", "Artwork Details"])

with search_tab:
    # Use reset counter to force widget reset by changing keys
    reset_counter = st.session_state.get("widget_reset_counter", 0)
    
    uploaded_file_combined = st.file_uploader(
        "Upload an artwork image",
        type=["jpg", "jpeg", "png"],
        key=f"combined_image_{reset_counter}"
    )

    text_query_combined = st.text_input(
        "Additional text description (optional)",
        placeholder="e.g., impressionist style, oil painting",
        key=f"combined_text_{reset_counter}"
    )

    top_k_combined = st.slider("Number of results", 1, 20, 5, key=f"combined_top_k_{reset_counter}")

    # Clear existing results when search inputs change
    current_input_signature = get_input_signature(uploaded_file_combined, text_query_combined)
    previous_signature = st.session_state.get("last_input_signature")
    if previous_signature is None:
        st.session_state["last_input_signature"] = current_input_signature
    elif current_input_signature != previous_signature:
        clear_search_results()
        st.session_state["last_input_signature"] = current_input_signature

    preview_image_bytes = None
    if uploaded_file_combined is not None:
        preview_image_bytes = uploaded_file_combined.getvalue()
        with BytesIO(preview_image_bytes) as preview_buffer:
            st.image(Image.open(preview_buffer), caption="Uploaded Image", use_container_width=True)
        uploaded_file_combined.seek(0)
    elif st.session_state.get("last_uploaded_image"):
        preview_image_bytes = st.session_state["last_uploaded_image"]
        with BytesIO(preview_image_bytes) as preview_buffer:
            st.image(Image.open(preview_buffer), caption="Last uploaded image", use_container_width=True)

    # Search and Reset buttons side by side
    col1, col2 = st.columns([3, 1])
    with col1:
        search_clicked = st.button("Search", key="combined_search_btn")
    with col2:
        reset_clicked = st.button("Reset", key="reset_btn", type="secondary")
    
    if reset_clicked:
        clear_search_results()
        st.session_state["last_input_signature"] = None
        st.session_state["last_uploaded_image"] = None
        st.session_state["last_text_query"] = ""
        # Increment counter to force widget reset by changing their keys
        st.session_state["widget_reset_counter"] = st.session_state.get("widget_reset_counter", 0) + 1
        st.rerun()

    if search_clicked:
        previous_search_id = st.session_state.get("current_search_id")
        if uploaded_file_combined is None and not text_query_combined:
            st.session_state["search_results"] = []
            st.session_state["last_uploaded_image"] = preview_image_bytes
            st.session_state["last_text_query"] = text_query_combined
            st.session_state["details_select_idx"] = 0
            st.session_state["last_search_feedback"] = ("warning", "Please upload an image, add a description, or both.")
            remove_search_state(previous_search_id)
            st.session_state["current_search_id"] = None
            for key in ("search_id", "selected_idx"):
                if key in st.query_params:
                    del st.query_params[key]
        else:
            using_uploaded_image = uploaded_file_combined is not None

            with st.spinner("Searching..."):
                results = []
                if using_uploaded_image and text_query_combined:
                    uploaded_file_combined.seek(0)
                    results = search_by_image_and_text(uploaded_file_combined, text_query_combined, top_k_combined)
                elif using_uploaded_image:
                    uploaded_file_combined.seek(0)
                    results = search_by_image(uploaded_file_combined, top_k_combined)
                else:
                    results = search_by_text(text_query_combined, top_k_combined)

            st.session_state["search_results"] = results if results else []
            st.session_state["last_uploaded_image"] = preview_image_bytes if using_uploaded_image else None
            st.session_state["last_text_query"] = text_query_combined

            if st.session_state["search_results"]:
                remove_search_state(previous_search_id)
                new_search_id = str(uuid.uuid4())
                st.session_state["current_search_id"] = new_search_id
                st.session_state["details_select_idx"] = 0
                st.session_state["last_search_feedback"] = ("success", f"Found {len(st.session_state['search_results'])} results. Click an artwork to view details.")
                st.query_params["search_id"] = new_search_id
                st.query_params["selected_idx"] = "0"
                persist_search_state()
            else:
                st.session_state["details_select_idx"] = 0
                st.session_state["last_search_feedback"] = ("info", "No results found.")
                remove_search_state(previous_search_id)
                st.session_state["current_search_id"] = None
                for key in ("search_id", "selected_idx"):
                    if key in st.query_params:
                        del st.query_params[key]

    feedback = st.session_state.get("last_search_feedback")
    if feedback:
        feedback_type, feedback_message = feedback
        feedback_method = getattr(st, feedback_type, st.info)
        feedback_method(feedback_message)

    stored_results = st.session_state.get("search_results", [])
    if stored_results:
        st.markdown("#### Search Results")
        cols = st.columns(min(3, len(stored_results)))
        current_search_id = st.session_state.get("current_search_id")
        for idx, result in enumerate(stored_results):
            with cols[idx % len(cols)]:
                display_search_result(result, idx, current_search_id)

        selected_idx = st.session_state.get("details_select_idx", 0)
        selected_idx = max(0, min(selected_idx, len(stored_results) - 1))
        st.session_state["details_select_idx"] = selected_idx
        persist_search_state()
    else:
        st.info("Upload an image or enter a description to start searching.")

with details_tab:
    st.markdown('<div id="artwork-details"></div>', unsafe_allow_html=True)
    st.header("Artwork Details")
    saved_results = st.session_state.get("search_results", [])

    if saved_results:
        current_idx = st.session_state.get("details_select_idx", 0)
        current_idx = max(0, min(current_idx, len(saved_results) - 1))

        selected_idx = st.selectbox(
            "Select an artwork to view details",
            options=list(range(len(saved_results))),
            format_func=lambda idx: f"{saved_results[idx].get('title', 'Unknown')} — {saved_results[idx].get('artist', 'Unknown')}",
            index=current_idx
        )

        if selected_idx != st.session_state.get("details_select_idx"):
            st.session_state["details_select_idx"] = selected_idx
            st.query_params["selected_idx"] = str(selected_idx)
            current_search_id = st.session_state.get("current_search_id")
            if current_search_id:
                st.query_params["search_id"] = current_search_id
            persist_search_state()

        selected_result = saved_results[selected_idx]

        display_artwork(selected_result)

        # Extract Palette section
        current_search_id = st.session_state.get("current_search_id")
        palette_key = f"{current_search_id}_{selected_idx}" if current_search_id else f"none_{selected_idx}"
        
        # Check if palette exists
        palette = st.session_state["color_palettes"].get(palette_key)
        
        # Button to extract palette - only show if not already extracted
        if not palette:
            if st.button("Extract Palette", key=f"extract_palette_{palette_key}", type="primary"):
                image_url = resolve_image_url(selected_result.get('image_url') or selected_result.get('image_path'))
                if image_url:
                    with st.spinner("Extracting color palette..."):
                        palette = extract_color_palette(image_url)
                        if palette:
                            st.session_state["color_palettes"][palette_key] = palette
                            st.session_state["active_tab_target"] = "details"
                            st.rerun()
                        else:
                            st.error("Error extracting palette")
                else:
                    st.warning("Image URL not available for palette extraction.")
        else:
            # Show button to re-extract if palette already exists
            if st.button("Re-extract Palette", key=f"re_extract_palette_{palette_key}"):
                image_url = resolve_image_url(selected_result.get('image_url') or selected_result.get('image_path'))
                if image_url:
                    with st.spinner("Extracting color palette..."):
                        new_palette = extract_color_palette(image_url)
                        if new_palette:
                            st.session_state["color_palettes"][palette_key] = new_palette
                            st.session_state["active_tab_target"] = "details"
                            st.rerun()
                        else:
                            st.error("Error extracting palette")
                else:
                    st.warning("Image URL not available for palette extraction.")
        
        # Get and display palette if available
        palette = st.session_state["color_palettes"].get(palette_key)
        if palette:
            st.markdown("---")
            display_color_palette(palette)

        last_text_query = st.session_state.get("last_text_query")
        if last_text_query:
            st.markdown(f"**Last description provided:** {last_text_query}")
    else:
        st.info("Run a search in the Search tab to view artwork details.")
