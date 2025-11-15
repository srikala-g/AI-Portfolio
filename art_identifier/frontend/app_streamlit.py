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


def display_artwork(result, is_uploaded_image=False):
    """Helper to display a single artwork card - image only."""
    if is_uploaded_image:
        # Display uploaded image from bytes
        if result.get('image_bytes'):
            try:
                with BytesIO(result['image_bytes']) as img_buffer:
                    st.image(Image.open(img_buffer), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display image: {e}")
    else:
        # Show image from API URL
        image_url = resolve_image_url(result.get('image_url') or result.get('image_path'))
        if image_url:
            try:
                st.image(image_url, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display image: {e}")


def display_search_result(result, idx: int, search_id: str | None):
    """Display search result card with non-clickable image only."""
    image_url = resolve_image_url(result.get('image_url') or result.get('image_path'))
    if image_url:
        st.image(image_url, use_container_width=True)
    else:
        st.info("Image preview unavailable.")


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


def get_artwork_backstory(artwork_id: str = None, image_bytes: bytes = None):
    """Fetch backstory for an artwork from the API."""
    try:
        if artwork_id:
            response = requests.get(
                f"{api_url}/artwork/{artwork_id}/backstory",
                timeout=60
            )
        elif image_bytes:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            response = requests.post(
                f"{api_url}/artwork/backstory-from-image",
                files=files,
                timeout=60
            )
        else:
            st.error("Either artwork_id or image_bytes must be provided")
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 503:
            st.error("Backstory generation is not available. Please set OPENAI_API_KEY in the backend.")
        elif e.response.status_code == 404:
            st.error(f"Artwork with ID {artwork_id} not found.")
        else:
            st.error(f"Error fetching backstory: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error fetching backstory: {str(e)}")
        return None


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

if "artwork_backstories" not in st.session_state:
    st.session_state["artwork_backstories"] = {}  # Store backstories by artwork_id

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
        # Handle both integer indices and "uploaded" string
        if selected_idx_param == "uploaded":
            st.session_state["details_select_idx"] = "uploaded"
        else:
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
                st.session_state["last_search_feedback"] = ("success", f"Found {len(st.session_state['search_results'])} results. Use the Artwork Details tab to view details.")
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
        # Only validate if it's an integer (not "uploaded" string)
        if isinstance(selected_idx, int):
            selected_idx = max(0, min(selected_idx, len(stored_results) - 1))
            st.session_state["details_select_idx"] = selected_idx
        persist_search_state()
    else:
        st.info("Upload an image or enter a description to start searching.")

with details_tab:
    st.markdown('<div id="artwork-details"></div>', unsafe_allow_html=True)
    st.header("Artwork Details")
    saved_results = st.session_state.get("search_results", [])
    uploaded_image_bytes = st.session_state.get("last_uploaded_image")

    # Build list of all artworks: uploaded image (if exists) + search results
    has_uploaded_image = uploaded_image_bytes is not None
    all_artworks = []
    
    if has_uploaded_image:
        all_artworks.append({
            "id": "uploaded",
            "type": "uploaded",
            "image_bytes": uploaded_image_bytes,
            "title": "Uploaded Image",
            "thumbnail_label": "📤 Uploaded"
        })
    
    for idx, result in enumerate(saved_results):
        all_artworks.append({
            "id": idx,
            "type": "search_result",
            "result": result,
            "thumbnail_label": f"{result.get('title', 'Unknown')[:20]}..."
        })

    if all_artworks:
        # Get current selection
        current_selection = st.session_state.get("details_select_idx", 0)
        
        # Display thumbnails in a row
        st.markdown("#### Select an artwork")
        num_thumbnails = len(all_artworks)
        cols = st.columns(num_thumbnails)
        
        for idx, artwork in enumerate(all_artworks):
            with cols[idx]:
                # Determine if this is the selected artwork
                is_selected = False
                if artwork["id"] == "uploaded" and current_selection == "uploaded":
                    is_selected = True
                elif isinstance(artwork["id"], int) and current_selection == artwork["id"]:
                    is_selected = True
                
                # Display thumbnail with border styling - make image clickable
                border_color = "#FF6B6B" if is_selected else "#E0E0E0"
                border_width = "3px" if is_selected else "2px"
                bg_color = "#FFF5F5" if is_selected else "white"
                
                # Build query parameters for selection
                selection_value = "uploaded" if artwork["id"] == "uploaded" else str(artwork["id"])
                current_search_id = st.session_state.get("current_search_id")
                query_params = f"selected_idx={selection_value}&active_tab=details"
                if current_search_id:
                    query_params += f"&search_id={current_search_id}"
                
                # Create a clickable container div
                container_id = f"thumb_{artwork['id']}"
                st.markdown(
                    f'<div id="{container_id}" style="border: {border_width} solid {border_color}; border-radius: 8px; padding: 4px; background: {bg_color}; margin-bottom: 8px; cursor: pointer;" onclick="window.location.href=\'?{query_params}#artwork-details\'">',
                    unsafe_allow_html=True
                )
                
                if artwork["type"] == "uploaded":
                    # Display uploaded image thumbnail
                    try:
                        with BytesIO(artwork["image_bytes"]) as img_buffer:
                            img = Image.open(img_buffer)
                            # Resize for thumbnail
                            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                            st.image(img, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    # Display search result thumbnail
                    result = artwork["result"]
                    image_url = resolve_image_url(result.get('image_url') or result.get('image_path'))
                    if image_url:
                        st.image(image_url, use_container_width=True)
                    else:
                        st.info("No image")
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Display selected artwork (selection is handled via onclick and query params)
        selected_result = None
        is_uploaded = False
        
        if current_selection == "uploaded" and has_uploaded_image:
            # Display uploaded image
            is_uploaded = True
            uploaded_result = {
                "image_bytes": uploaded_image_bytes,
                "title": "Uploaded Image",
                "type": "uploaded"
            }
            display_artwork(uploaded_result, is_uploaded_image=True)
            selected_result = uploaded_result
        elif isinstance(current_selection, int) and 0 <= current_selection < len(saved_results):
            # Display search result
            selected_result = saved_results[current_selection]
            display_artwork(selected_result, is_uploaded_image=False)

        st.markdown("---")
        
        # Prepare keys for backstory and palette (only for search results, not uploaded images)
        artwork_id = selected_result.get("id") if selected_result and not is_uploaded else None
        selection_key = "uploaded" if is_uploaded else (artwork_id if artwork_id else str(current_selection))
        backstory_key = f"backstory_{selection_key}"
        current_search_id = st.session_state.get("current_search_id")
        palette_key = f"palette_{current_search_id}_{selection_key}" if current_search_id else f"palette_{selection_key}"
        
        # Check if backstory and palette exist
        backstory_data = st.session_state["artwork_backstories"].get(backstory_key)
        palette = st.session_state["color_palettes"].get(palette_key)
        
        # Place both buttons side by side
        col1, col2 = st.columns(2)
        
        with col1:
            # Backstory button
            if not backstory_data:
                if st.button("Generate Backstory", key=f"generate_backstory_{backstory_key}", type="primary", use_container_width=True):
                    with st.spinner("Generating backstory using AI..."):
                        if is_uploaded and uploaded_image_bytes:
                            # Generate backstory from uploaded image
                            backstory_result = get_artwork_backstory(image_bytes=uploaded_image_bytes)
                        elif artwork_id:
                            # Generate backstory from artwork ID
                            backstory_result = get_artwork_backstory(artwork_id=artwork_id)
                        else:
                            st.warning("No artwork or image available for backstory generation.")
                            backstory_result = None
                        
                        if backstory_result:
                            st.session_state["artwork_backstories"][backstory_key] = backstory_result
                            st.session_state["active_tab_target"] = "details"
                            st.rerun()
            else:
                if st.button("Regenerate Backstory", key=f"regenerate_backstory_{backstory_key}", use_container_width=True):
                    with st.spinner("Regenerating backstory using AI..."):
                        if is_uploaded and uploaded_image_bytes:
                            # Regenerate backstory from uploaded image
                            backstory_result = get_artwork_backstory(image_bytes=uploaded_image_bytes)
                        elif artwork_id:
                            # Regenerate backstory from artwork ID
                            backstory_result = get_artwork_backstory(artwork_id=artwork_id)
                        else:
                            st.warning("No artwork or image available for backstory generation.")
                            backstory_result = None
                        
                        if backstory_result:
                            st.session_state["artwork_backstories"][backstory_key] = backstory_result
                            st.session_state["active_tab_target"] = "details"
                            st.rerun()
        
        with col2:
            # Palette button
            if not palette:
                if st.button("Extract Palette", key=f"extract_palette_{palette_key}", type="primary", use_container_width=True):
                    if is_uploaded and uploaded_image_bytes:
                        # Extract palette from uploaded image bytes
                        with st.spinner("Extracting color palette..."):
                            try:
                                # Use PIL to open from bytes directly
                                img = Image.open(BytesIO(uploaded_image_bytes))
                                # Convert to RGB if needed
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                # Extract palette directly from PIL Image
                                # Use the same logic as extract_color_palette but with PIL Image
                                import numpy as np
                                max_size = 400
                                if img.width > max_size or img.height > max_size:
                                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                
                                img_array = np.array(img)
                                pixels = img_array.reshape(-1, 3).astype(np.float32)
                                
                                # Use the same color extraction logic
                                brightness = pixels.max(axis=1)
                                saturation = (pixels.max(axis=1) - pixels.min(axis=1)) / (pixels.max(axis=1) + 1e-6)
                                
                                bright_saturated = (brightness > 180) & (saturation > 0.3)
                                bright_unsaturated = (brightness > 180) & (saturation <= 0.3)
                                mid_saturated = (brightness >= 80) & (brightness <= 180) & (saturation > 0.2)
                                mid_unsaturated = (brightness >= 80) & (brightness <= 180) & (saturation <= 0.2)
                                dark = brightness < 80
                                
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
                                    sample_size = min(10000, len(pixels))
                                    indices = np.random.choice(len(pixels), sample_size, replace=False)
                                    pixels_to_cluster = pixels[indices]
                                
                                num_colors = 12
                                centroids = []
                                
                                if bright_saturated.any():
                                    bright_sat_pixels = pixels[bright_saturated]
                                    idx = np.random.randint(0, len(bright_sat_pixels))
                                    centroids.append(bright_sat_pixels[idx].copy())
                                
                                if mid_saturated.any() and len(centroids) < num_colors:
                                    mid_sat_pixels = pixels[mid_saturated]
                                    idx = np.random.randint(0, len(mid_sat_pixels))
                                    centroids.append(mid_sat_pixels[idx].copy())
                                
                                remaining = num_colors - len(centroids)
                                
                                if remaining > 0:
                                    if len(centroids) == 0:
                                        first_idx = np.random.randint(0, len(pixels_to_cluster))
                                        centroids.append(pixels_to_cluster[first_idx].copy())
                                        remaining -= 1
                                    
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
                                    distances = np.array([
                                        np.linalg.norm(pixels_to_cluster - centroid, axis=1)
                                        for centroid in centroids
                                    ])
                                    assignments = np.argmin(distances, axis=0)
                                    
                                    new_centroids = np.array([
                                        pixels_to_cluster[assignments == i].mean(axis=0) if np.any(assignments == i) 
                                        else centroids[i]
                                        for i in range(num_colors)
                                    ])
                                    
                                    if np.allclose(centroids, new_centroids, atol=1.0):
                                        break
                                    
                                    centroids = new_centroids
                                
                                # Extract final palette
                                palette = []
                                seen_colors = set()
                                
                                for i in range(num_colors):
                                    r, g, b = centroids[i]
                                    r, g, b = int(round(r)), int(round(g)), int(round(b))
                                    r = max(0, min(255, r))
                                    g = max(0, min(255, g))
                                    b = max(0, min(255, b))
                                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                                    
                                    if hex_color not in seen_colors:
                                        palette.append(hex_color)
                                        seen_colors.add(hex_color)
                                
                                if palette:
                                    st.session_state["color_palettes"][palette_key] = palette
                                    st.session_state["active_tab_target"] = "details"
                                    st.rerun()
                                else:
                                    st.error("Error extracting palette")
                            except Exception as e:
                                st.error(f"Error extracting palette: {str(e)}")
                    else:
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
                if st.button("Re-extract Palette", key=f"re_extract_palette_{palette_key}", use_container_width=True):
                    if is_uploaded and uploaded_image_bytes:
                        # Re-extract palette from uploaded image bytes (same logic as above)
                        with st.spinner("Extracting color palette..."):
                            try:
                                img = Image.open(BytesIO(uploaded_image_bytes))
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                
                                import numpy as np
                                max_size = 400
                                if img.width > max_size or img.height > max_size:
                                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                
                                img_array = np.array(img)
                                pixels = img_array.reshape(-1, 3).astype(np.float32)
                                brightness = pixels.max(axis=1)
                                saturation = (pixels.max(axis=1) - pixels.min(axis=1)) / (pixels.max(axis=1) + 1e-6)
                                
                                bright_saturated = (brightness > 180) & (saturation > 0.3)
                                mid_saturated = (brightness >= 80) & (brightness <= 180) & (saturation > 0.2)
                                
                                samples_per_region = 2000
                                sampled_pixels = []
                                for mask in [bright_saturated, (brightness > 180) & (saturation <= 0.3), mid_saturated, (brightness >= 80) & (brightness <= 180) & (saturation <= 0.2), brightness < 80]:
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
                                    sample_size = min(10000, len(pixels))
                                    indices = np.random.choice(len(pixels), sample_size, replace=False)
                                    pixels_to_cluster = pixels[indices]
                                
                                num_colors = 12
                                centroids = []
                                if bright_saturated.any():
                                    bright_sat_pixels = pixels[bright_saturated]
                                    idx = np.random.randint(0, len(bright_sat_pixels))
                                    centroids.append(bright_sat_pixels[idx].copy())
                                if mid_saturated.any() and len(centroids) < num_colors:
                                    mid_sat_pixels = pixels[mid_saturated]
                                    idx = np.random.randint(0, len(mid_sat_pixels))
                                    centroids.append(mid_sat_pixels[idx].copy())
                                
                                remaining = num_colors - len(centroids)
                                if remaining > 0:
                                    if len(centroids) == 0:
                                        first_idx = np.random.randint(0, len(pixels_to_cluster))
                                        centroids.append(pixels_to_cluster[first_idx].copy())
                                        remaining -= 1
                                    for _ in range(remaining):
                                        distances = np.array([min(np.linalg.norm(pixel - centroid) for centroid in centroids) for pixel in pixels_to_cluster])
                                        probabilities = distances ** 2
                                        probabilities /= probabilities.sum()
                                        next_idx = np.random.choice(len(pixels_to_cluster), p=probabilities)
                                        centroids.append(pixels_to_cluster[next_idx].copy())
                                
                                centroids = np.array(centroids)
                                max_iterations = 30
                                for iteration in range(max_iterations):
                                    distances = np.array([np.linalg.norm(pixels_to_cluster - centroid, axis=1) for centroid in centroids])
                                    assignments = np.argmin(distances, axis=0)
                                    new_centroids = np.array([pixels_to_cluster[assignments == i].mean(axis=0) if np.any(assignments == i) else centroids[i] for i in range(num_colors)])
                                    if np.allclose(centroids, new_centroids, atol=1.0):
                                        break
                                    centroids = new_centroids
                                
                                palette = []
                                seen_colors = set()
                                for i in range(num_colors):
                                    r, g, b = centroids[i]
                                    r, g, b = int(round(r)), int(round(g)), int(round(b))
                                    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
                                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                                    if hex_color not in seen_colors:
                                        palette.append(hex_color)
                                        seen_colors.add(hex_color)
                                
                                if palette:
                                    st.session_state["color_palettes"][palette_key] = palette
                                    st.session_state["active_tab_target"] = "details"
                                    st.rerun()
                                else:
                                    st.error("Error extracting palette")
                            except Exception as e:
                                st.error(f"Error extracting palette: {str(e)}")
                    else:
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
        
        # Display backstory if available
        if backstory_data:
            st.markdown("---")
            st.markdown("#### Artwork Information")
            # Display all metadata from backstory
            title = backstory_data.get("title")
            artist = backstory_data.get("artist")
            genre = backstory_data.get("genre")
            style = backstory_data.get("style")
            style_features = backstory_data.get("style_features", [])
            
            if title:
                st.markdown(f"**Title:** {title}")
            if artist:
                st.markdown(f"**Artist:** {artist}")
            if genre:
                st.markdown(f"**Genre:** {genre}")
            if style:
                st.markdown(f"**Style:** {style}")
            
            # Display salient style features in an expandable section
            if style_features:
                with st.expander("**Salient Style Features**", expanded=False):
                    if isinstance(style_features, list):
                        for feature in style_features:
                            st.markdown(f"• {feature}")
                    else:
                        st.markdown(f"• {style_features}")
            
            st.markdown("---")
            st.markdown("#### Backstory")
            st.markdown(backstory_data.get("backstory", ""))
            if backstory_data.get("is_image_analysis"):
                st.caption("Generated from image analysis")
            else:
                st.caption("Generated using artwork metadata and similar artworks")
        
        # Display palette if available
        if palette:
            st.markdown("---")
            display_color_palette(palette)
    else:
        st.info("Run a search in the Search tab to view artwork details.")
