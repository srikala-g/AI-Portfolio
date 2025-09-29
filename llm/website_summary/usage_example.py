#!/usr/bin/env python3
"""
Usage examples for the Website Summary Tool

This file demonstrates different ways to use the website_summary.py module.
"""

from website_summary import WebsiteSummarizer, Website, WebsiteSelenium

def example_basic_usage():
    """Basic usage example."""
    print("=== BASIC USAGE EXAMPLE ===")
    
    # Initialize the summarizer
    summarizer = WebsiteSummarizer()
    
    # Summarize a single URL
    url = "https://edwarddonner.com"
    print(f"Summarizing: {url}")
    
    try:
        summary = summarizer.summarize(url)
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"Error: {e}")

def example_selenium_usage():
    """Selenium usage example."""
    print("\n=== SELENIUM USAGE EXAMPLE ===")
    
    summarizer = WebsiteSummarizer()
    url = "https://openai.com"  # JavaScript-heavy site
    
    print(f"Summarizing with Selenium: {url}")
    
    try:
        summary = summarizer.summarize_with_selenium(url, headless=True)
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"Error: {e}")

def example_auto_mode():
    """Auto mode usage example."""
    print("\n=== AUTO MODE EXAMPLE ===")
    
    summarizer = WebsiteSummarizer()
    urls = [
        "https://edwarddonner.com",  # Should use HTTP
        "https://openai.com",        # Should use Selenium
    ]
    
    for url in urls:
        print(f"\nAuto-summarizing: {url}")
        try:
            summary = summarizer.summarize_auto(url)
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"Error: {e}")

def example_direct_classes():
    """Direct class usage example."""
    print("\n=== DIRECT CLASS USAGE EXAMPLE ===")
    
    # Using Website class directly
    url = "https://cnn.com"
    print(f"Scraping with Website class: {url}")
    
    try:
        website = Website(url)
        print(f"Title: {website.title}")
        print(f"Content length: {len(website.text)} characters")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Website Summary Tool - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_selenium_usage()
    example_auto_mode()
    example_direct_classes()
    
    print("\n" + "=" * 50)
    print("For interactive usage, run: python website_summary.py")
