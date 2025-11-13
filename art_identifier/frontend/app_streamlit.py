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

active_tab_param = _first_query_value(incoming_query_params.get("active_tab"))
if active_tab_param == "details":
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
    if "active_tab" in st.query_params:
        del st.query_params["active_tab"]


# Check API health
is_healthy, artworks_count = check_api_health()
if not is_healthy:
    st.error(f"⚠️ API is not available at {api_url}. Please make sure the backend is running.")
    st.stop()
search_tab, details_tab = st.tabs(["Search", "Artwork Details"])

with search_tab:
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

    search_clicked = st.button("Search", key="combined_search_btn")

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
            with st.spinner("Searching..."):
                results = []
                if uploaded_file_combined is not None and text_query_combined:
                    uploaded_file_combined.seek(0)
                    results = search_by_image_and_text(uploaded_file_combined, text_query_combined, top_k_combined)
                elif uploaded_file_combined is not None:
                    uploaded_file_combined.seek(0)
                    results = search_by_image(uploaded_file_combined, top_k_combined)
                else:
                    results = search_by_text(text_query_combined, top_k_combined)

            st.session_state["search_results"] = results if results else []
            st.session_state["last_uploaded_image"] = preview_image_bytes
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

        last_text_query = st.session_state.get("last_text_query")
        if last_text_query:
            st.markdown(f"**Last description provided:** {last_text_query}")
    else:
        st.info("Run a search in the Search tab to view artwork details.")
