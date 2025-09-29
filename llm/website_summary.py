"""
Website Summary Tool
This module provides functionality to scrape websites and generate AI-powered summaries.
Supports both basic HTTP scraping and JavaScript-rendered websites using Selenium.
"""

import os
import requests
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# Selenium imports (optional - only needed for JavaScript websites)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Install with: pip install selenium")

# Gradio imports
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None  # Set gr to None when not available
    print("Gradio not available. Install with: pip install gradio")


class Website:
    """
    A class to represent a Webpage with scraping capabilities.
    """
    
    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library.
        
        Args:
            url (str): The URL of the website to scrape
        """
        self.url = url
        
        # Some websites need you to use proper headers when fetching them:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        self.title = soup.title.string if soup.title else "No title found"
        
        # Remove irrelevant elements
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
            
        self.text = soup.body.get_text(separator="\n", strip=True)


class WebsiteSelenium:
    """
    A class to represent a Webpage with Selenium-based scraping capabilities.
    This handles JavaScript-rendered websites that basic HTTP requests cannot.
    """
    
    def __init__(self, url, headless=True, wait_time=10):
        """
        Create this Website object using Selenium WebDriver.
        
        Args:
            url (str): The URL of the website to scrape
            headless (bool): Whether to run browser in headless mode
            wait_time (int): Maximum time to wait for page elements to load
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not installed. Install with: pip install selenium")
        
        self.url = url
        self.driver = None
        
        try:
            # Set up Chrome options
            chrome_options = Options()
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36")
            
            # Initialize the driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page title
            self.title = self.driver.title if self.driver.title else "No title found"
            
            # Get page content
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Remove irrelevant elements
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
                
            self.text = soup.body.get_text(separator="\n", strip=True)
            
        except (TimeoutException, WebDriverException) as e:
            print(f"Error loading {url}: {e}")
            self.title = "Error loading page"
            self.text = f"Failed to load content from {url}"
        finally:
            if self.driver:
                self.driver.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()


class WebsiteSummarizer:
    """
    A class to handle website summarization using OpenAI.
    """
    
    def __init__(self):
        """Initialize the summarizer with OpenAI client."""
        # Load environment variables
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("No API key was found - please check your .env file!")
        elif not api_key.startswith("sk-proj-"):
            raise ValueError("API key doesn't start with 'sk-proj-' - please check your key!")
        elif api_key.strip() != api_key:
            raise ValueError("API key has whitespace - please remove spaces/tabs!")
        
        self.openai = OpenAI()
        self.system_prompt = ("You are an assistant that analyzes the contents of a website "
                             "and provides a short summary, ignoring text that might be navigation related. "
                             "Respond in markdown.")
    
    def user_prompt_for(self, website):
        """
        Create a user prompt for website summarization.
        
        Args:
            website (Website): The website object to summarize
            
        Returns:
            str: The user prompt for the AI
        """
        user_prompt = f"You are looking at a website titled {website.title}"
        user_prompt += "\nThe contents of this website is as follows; "
        user_prompt += "please provide a short summary of this website in markdown. "
        user_prompt += "If it includes news or announcements, then summarize these too.\n\n"
        user_prompt += website.text
        return user_prompt
    
    def messages_for(self, website):
        """
        Create the messages structure for OpenAI API.
        
        Args:
            website (Website): The website object to summarize
            
        Returns:
            list: List of message dictionaries for OpenAI API
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt_for(website)}
        ]
    
    def summarize(self, url):
        """
        Summarize a website given its URL using basic HTTP scraping.
        
        Args:
            url (str): The URL of the website to summarize
            
        Returns:
            str: The AI-generated summary of the website
        """
        website = Website(url)
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages_for(website)
        )
        return response.choices[0].message.content
    
    def summarize_with_selenium(self, url, headless=True, wait_time=10):
        """
        Summarize a JavaScript-rendered website using Selenium.
        
        Args:
            url (str): The URL of the website to summarize
            headless (bool): Whether to run browser in headless mode
            wait_time (int): Maximum time to wait for page elements to load
            
        Returns:
            str: The AI-generated summary of the website
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not installed. Install with: pip install selenium")
        
        website = WebsiteSelenium(url, headless=headless, wait_time=wait_time)
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages_for(website)
        )
        return response.choices[0].message.content
    
    def summarize_auto(self, url, try_selenium_first=False):
        """
        Automatically choose the best scraping method for a website.
        
        Args:
            url (str): The URL of the website to summarize
            try_selenium_first (bool): Whether to try Selenium first, then fall back to HTTP
            
        Returns:
            str: The AI-generated summary of the website
        """
        if try_selenium_first and SELENIUM_AVAILABLE:
            try:
                print(f"Trying Selenium for {url}...")
                return self.summarize_with_selenium(url)
            except Exception as e:
                print(f"Selenium failed for {url}: {e}")
                print(f"Falling back to HTTP scraping...")
                return self.summarize(url)
        else:
            try:
                print(f"Trying HTTP scraping for {url}...")
                return self.summarize(url)
            except Exception as e:
                print(f"HTTP scraping failed for {url}: {e}")
                if SELENIUM_AVAILABLE:
                    print(f"Falling back to Selenium...")
                    return self.summarize_with_selenium(url)
                else:
                    raise e


def get_user_choice():
    """
    Get user input for scraping method choice.
    
    Returns:
        str: User's choice ('selenium', 'http', 'auto', or 'interactive')
    """
    print("\n" + "="*60)
    print("WEBSITE SUMMARY TOOL")
    print("="*60)
    print("Choose your scraping method:")
    print("1. HTTP scraping (fast, works for most static websites)")
    print("2. Selenium scraping (slower, works for JavaScript websites)")
    print("3. Auto mode (tries HTTP first, falls back to Selenium)")
    print("4. Interactive mode (ask for each website)")
    print("5. Demo mode (test with predefined URLs)")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            return "http"
        elif choice == "2":
            return "selenium"
        elif choice == "3":
            return "auto"
        elif choice == "4":
            return "interactive"
        elif choice == "5":
            return "demo"
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def get_urls_from_user():
    """
    Get URLs from user input.
    
    Returns:
        list: List of URLs to scrape
    """
    urls = []
    print("\nEnter URLs to summarize (one per line, empty line to finish):")
    
    while True:
        url = input("URL: ").strip()
        if not url:
            break
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        urls.append(url)
    
    return urls


def summarize_with_choice(summarizer, url, method):
    """
    Summarize a URL using the specified method.
    
    Args:
        summarizer: WebsiteSummarizer instance
        url (str): URL to summarize
        method (str): Method to use ('http', 'selenium', 'auto')
        
    Returns:
        str: Summary of the website
    """
    try:
        if method == "http":
            return summarizer.summarize(url)
        elif method == "selenium":
            if not SELENIUM_AVAILABLE:
                raise ImportError("Selenium not available. Install with: pip install selenium")
            return summarizer.summarize_with_selenium(url)
        elif method == "auto":
            return summarizer.summarize_auto(url)
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        raise Exception(f"Failed to summarize {url}: {e}")


def main():
    """
    Interactive website summarization tool with user choice.
    """
    try:
        summarizer = WebsiteSummarizer()
        
        # Get user's choice
        choice = get_user_choice()
        
        if choice == "demo":
            # Demo mode with predefined URLs
            urls = [
                "https://cnn.com",          # Should work with HTTP
                "https://openai.com",       # May need Selenium (JavaScript-heavy)
                "https://anthropic.com"     # Should work with HTTP
            ]
            method = "auto"
        else:
            # Get URLs from user
            urls = get_urls_from_user()
            if not urls:
                print("No URLs provided. Exiting.")
                return
            method = choice
        
        # Process each URL
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"Processing URL {i}/{len(urls)}: {url}")
            print(f"{'='*60}")
            
            try:
                if choice == "interactive":
                    # Ask for each URL individually
                    print(f"\nChoose method for: {url}")
                    print("1. HTTP scraping")
                    print("2. Selenium scraping")
                    print("3. Auto mode")
                    
                    while True:
                        url_choice = input("Enter choice (1-3): ").strip()
                        if url_choice == "1":
                            url_method = "http"
                            break
                        elif url_choice == "2":
                            url_method = "selenium"
                            break
                        elif url_choice == "3":
                            url_method = "auto"
                            break
                        else:
                            print("Invalid choice. Please enter 1, 2, or 3.")
                else:
                    url_method = method
                
                # Summarize the URL
                summary = summarize_with_choice(summarizer, url, url_method)
                print(f"\nSummary:")
                print("-" * 40)
                print(summary)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
            
            # Ask if user wants to continue (for multiple URLs)
            if i < len(urls):
                continue_choice = input(f"\nContinue to next URL? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes', '']:
                    break
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your .env file and ensure OPENAI_API_KEY is set correctly.")


def demo_selenium_only():
    """
    Demo function showing Selenium-only scraping for JavaScript websites.
    """
    if not SELENIUM_AVAILABLE:
        print("Selenium not available. Install with: pip install selenium")
        return
    
    try:
        summarizer = WebsiteSummarizer()
        
        # JavaScript-heavy websites that need Selenium
        js_urls = [
            "https://openai.com",
            "https://reactjs.org",
            "https://vuejs.org"
        ]
        
        for url in js_urls:
            print(f"\n{'='*50}")
            print(f"Selenium scraping: {url}")
            print(f"{'='*50}")
            
            try:
                summary = summarizer.summarize_with_selenium(url, headless=True)
                print(summary)
            except Exception as e:
                print(f"Selenium failed for {url}: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your .env file and ensure OPENAI_API_KEY is set correctly.")


def gradio_summarize(url, method, headless=True):
    """
    Gradio interface function for website summarization.
    
    Args:
        url (str): URL to summarize
        method (str): Scraping method ('http', 'selenium', 'auto')
        headless (bool): Whether to run Selenium in headless mode
        
    Returns:
        str: Summary of the website or error message
    """
    if not url.strip():
        return "Please enter a URL to summarize."
    
    # Add https:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        summarizer = WebsiteSummarizer()
        
        if method == "HTTP Scraping":
            summary = summarizer.summarize(url)
        elif method == "Selenium Scraping":
            if not SELENIUM_AVAILABLE:
                return "Selenium not available. Please install with: pip install selenium"
            summary = summarizer.summarize_with_selenium(url, headless=headless)
        elif method == "Auto Mode":
            summary = summarizer.summarize_auto(url)
        else:
            return f"Unknown method: {method}"
        
        return f"**Summary of {url}:**\n\n{summary}"
        
    except Exception as e:
        return f"Error: {str(e)}"


def create_gradio_interface():
    """
    Create and return a Gradio interface for the website summarizer.
    
    Returns:
        gr.Interface: Gradio interface object
    """
    if not GRADIO_AVAILABLE or gr is None:
        raise ImportError("Gradio not available. Install with: pip install gradio")
    
    # Define the interface
    interface = gr.Interface(
        fn=gradio_summarize,
        inputs=[
            gr.Textbox(
                label="Website URL",
                placeholder="Enter a URL to summarize (e.g., https://example.com)",
                lines=1
            ),
            gr.Dropdown(
                choices=["HTTP Scraping", "Selenium Scraping", "Auto Mode"],
                value="Auto Mode",
                label="Scraping Method",
                info="Choose how to scrape the website"
            ),
            gr.Checkbox(
                value=True,
                label="Headless Mode (Selenium)",
                info="Run browser in background (recommended)"
            )
        ],
        outputs=gr.Markdown(
            label="Website Summary",
            show_copy_button=True
        ),
        title="🌐 Website Summary Tool",
        description="""
        **AI-powered website summarization tool** that can handle both static and JavaScript-rendered websites.
        
        **Features:**
        - 📄 **HTTP Scraping**: Fast scraping for static websites
        - 🤖 **Selenium Scraping**: Handles JavaScript-rendered websites
        - 🔄 **Auto Mode**: Intelligently chooses the best method
        
        **Usage:**
        1. Enter a website URL
        2. Choose your scraping method
        3. Click "Submit" to get an AI-generated summary
        """,
        examples=[
            ["https://edwarddonner.com", "Auto Mode", True],
            ["https://openai.com", "Selenium Scraping", True],
            ["https://cnn.com", "HTTP Scraping", True],
            ["https://anthropic.com", "Auto Mode", True]
        ],
        cache_examples=False,
        theme="soft",
        allow_flagging="never"
    )
    
    return interface


def launch_gradio_app():
    """
    Launch the Gradio web interface.
    """
    if not GRADIO_AVAILABLE or gr is None:
        print("Gradio not available. Install with: pip install gradio")
        return
    
    try:
        # Import gradio here to ensure it's available
        import gradio as gr
        interface = create_gradio_interface()
        print("🚀 Launching Gradio interface...")
        print("📱 Open your browser and navigate to the local URL shown below")
        interface.launch(
            share=False,  # Set to True to create a public link
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,  # Default Gradio port
            show_error=True
        )
    except Exception as e:
        print(f"Error launching Gradio interface: {e}")


if __name__ == "__main__":
    import sys
    
    # Check if user wants Gradio interface
    if len(sys.argv) > 1 and sys.argv[1] == "--gradio":
        launch_gradio_app()
    else:
        # Default to command-line interface
        main()
        # Uncomment the line below to test Selenium-only scraping
        # demo_selenium_only()
