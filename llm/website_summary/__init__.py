"""
Website Summary Tool Package

A comprehensive tool for AI-powered website summarization using OpenAI and Ollama models.
Supports both HTTP scraping and Selenium-based JavaScript website handling.
"""

from .website_summary import WebsiteSummarizer, Website, WebsiteSelenium
from .gradio_app import create_interface, launch_gradio_app

__version__ = "1.0.0"
__author__ = "AI Portfolio"

__all__ = [
    "WebsiteSummarizer",
    "Website", 
    "WebsiteSelenium",
    "create_interface",
    "launch_gradio_app"
]
