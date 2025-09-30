"""
Brochure Generator - AI-powered company brochure generation from website analysis.

This module extracts code from day5.ipynb to create professional company brochures
by analyzing website content and generating marketing materials.
"""

import os
import requests
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI


class Website:
    """
    A utility class to represent a Website that we have scraped, now with links.
    """
    
    def __init__(self, url: str):
        """
        Initialize website object with content and links.
        
        Args:
            url (str): The URL of the website to scrape
        """
        self.url = url
        
        # Some websites need you to use proper headers when fetching them:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        
        self.title = soup.title.string if soup.title else "No title found"
        
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        
        # Extract links
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self) -> str:
        """Get formatted website contents."""
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


class BrochureGenerator:
    """
    A class to generate company brochures from website analysis.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the brochure generator.
        
        Args:
            model (str): OpenAI model to use for generation
        """
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
            print("API key looks good so far")
        else:
            print("There might be a problem with your API key? Please check your .env file!")
        
        self.model = model
        self.openai = OpenAI()
        
        # System prompt for link analysis with multi-shot prompting
        self.link_system_prompt = """You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.

You should respond in JSON format. Here are examples of good link analysis:

Example 1 - Tech Company:
Input links: ["/about", "/careers", "/products", "/contact", "/privacy", "/terms"]
Output: {
    "links": [
        {"type": "about page", "url": "https://company.com/about"},
        {"type": "careers page", "url": "https://company.com/careers"},
        {"type": "products page", "url": "https://company.com/products"}
    ]
}

Example 2 - Art Gallery:
Input links: ["/gallery", "/artists", "/exhibitions", "/contact", "/news", "/shop"]
Output: {
    "links": [
        {"type": "gallery page", "url": "https://gallery.com/gallery"},
        {"type": "artists page", "url": "https://gallery.com/artists"},
        {"type": "exhibitions page", "url": "https://gallery.com/exhibitions"}
    ]
}

Example 3 - Service Company:
Input links: ["/services", "/team", "/portfolio", "/testimonials", "/blog", "/faq"]
Output: {
    "links": [
        {"type": "services page", "url": "https://company.com/services"},
        {"type": "team page", "url": "https://company.com/team"},
        {"type": "portfolio page", "url": "https://company.com/portfolio"},
        {"type": "testimonials page", "url": "https://company.com/testimonials"}
    ]
}

Now analyze the provided links and respond in the same JSON format."""
        
        # System prompt for brochure generation with multi-shot prompting
        self.brochure_system_prompt = """You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.

Here are examples of excellent brochure formats:

Example 1 - Tech Company Brochure:
# [Company Name] - Innovation in Technology

## About Us
[Company] is a leading technology company specializing in [specific technology/solution]. Founded in [year], we have been at the forefront of [industry] innovation.

## Our Mission
To [mission statement] through cutting-edge technology and exceptional service.

## What We Do
- **Core Services**: [List main services]
- **Technology Stack**: [List technologies used]
- **Target Market**: [Describe target customers]

## Why Choose Us
- **Expertise**: [Years] of experience in [field]
- **Innovation**: [Specific innovations or achievements]
- **Support**: [Customer support details]

## Career Opportunities
We're always looking for talented individuals to join our team. Current openings include:
- [Job title 1]: [Brief description]
- [Job title 2]: [Brief description]

## Contact Information
- Website: [website]
- Email: [email]
- Phone: [phone]

---

Example 2 - Art Gallery Brochure:
# [Gallery Name] - Contemporary Art Gallery

## Welcome to [Gallery Name]
[Gallery Name] is a premier contemporary art gallery showcasing works by emerging and established artists.

## Our Collection
Our gallery features:
- **Contemporary Art**: Modern works by local and international artists
- **Exhibitions**: Regular rotating exhibitions
- **Artist Spotlights**: Featured artist programs

## Current Exhibition
**[Exhibition Name]**
[Brief description of current exhibition]

## Gallery Information
- **Location**: [Address]
- **Hours**: [Operating hours]
- **Admission**: [Pricing information]

## Visit Us
Experience the beauty of contemporary art at [Gallery Name].

---

Example 3 - Service Company Brochure:
# [Company Name] - Professional Services

## About [Company Name]
[Company Name] has been providing [services] for [number] years, serving clients across [geographic area].

## Our Services
- **[Service 1]**: [Description]
- **[Service 2]**: [Description]
- **[Service 3]**: [Description]

## Our Team
Our experienced professionals bring [X] years of combined experience to every project.

## Client Testimonials
"[Testimonial quote]" - [Client Name]

## Get Started
Contact us today to discuss your project needs.

---

Now create a brochure following these examples, adapting the format to the specific company and information provided."""
    
    def get_links_user_prompt(self, website: Website) -> str:
        """
        Create user prompt for link analysis.
        
        Args:
            website: Website object with links
            
        Returns:
            str: User prompt for link analysis
        """
        user_prompt = f"Here is the list of links on the website of {website.url} - "
        user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. "
        user_prompt += "Do not include Terms of Service, Privacy, email links.\n"
        user_prompt += "Links (some might be relative links):\n"
        user_prompt += "\n".join(website.links)
        return user_prompt
    
    def get_links(self, url: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Get relevant links from a website using AI analysis.
        
        Args:
            url (str): Website URL to analyze
            
        Returns:
            Dict: JSON response with relevant links
        """
        website = Website(url)
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.link_system_prompt},
                {"role": "user", "content": self.get_links_user_prompt(website)}
            ],
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        return json.loads(result)
    
    def get_all_details(self, url: str) -> str:
        """
        Get all relevant details from a website and its linked pages.
        
        Args:
            url (str): Website URL to analyze
            
        Returns:
            str: Combined content from main page and relevant links
        """
        result = "Landing page:\n"
        result += Website(url).get_contents()
        
        links = self.get_links(url)
        print("Found links:", links)
        
        for link in links["links"]:
            try:
                result += f"\n\n{link['type']}\n"
                result += Website(link["url"]).get_contents()
            except Exception as e:
                print(f"Error scraping {link['url']}: {e}")
                continue
        
        return result
    
    def get_brochure_user_prompt(self, company_name: str, url: str) -> str:
        """
        Create user prompt for brochure generation.
        
        Args:
            company_name (str): Name of the company
            url (str): Company website URL
            
        Returns:
            str: User prompt for brochure generation
        """
        user_prompt = f"You are looking at a company called: {company_name}\n"
        user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
        user_prompt += self.get_all_details(url)
        user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
        return user_prompt
    
    def create_brochure(self, company_name: str, url: str) -> str:
        """
        Create a brochure for a company.
        
        Args:
            company_name (str): Name of the company
            url (str): Company website URL
            
        Returns:
            str: Generated brochure in markdown format
        """
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.brochure_system_prompt},
                {"role": "user", "content": self.get_brochure_user_prompt(company_name, url)}
            ],
        )
        result = response.choices[0].message.content
        return result
    
    def stream_brochure(self, company_name: str, url: str):
        """
        Create a brochure with streaming output.
        
        Args:
            company_name (str): Name of the company
            url (str): Company website URL
            
        Yields:
            str: Chunks of the generated brochure
        """
        stream = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.brochure_system_prompt},
                {"role": "user", "content": self.get_brochure_user_prompt(company_name, url)}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def set_humorous_tone(self):
        """Set the system prompt to create humorous brochures."""
        self.brochure_system_prompt = """You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.
Include details of company culture, customers and careers/jobs if you have the information."""
    
    def set_professional_tone(self):
        """Set the system prompt to create professional brochures."""
        self.brochure_system_prompt = """You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.

Here are examples of excellent brochure formats:

Example 1 - Tech Company Brochure:
# [Company Name] - Innovation in Technology

## About Us
[Company] is a leading technology company specializing in [specific technology/solution]. Founded in [year], we have been at the forefront of [industry] innovation.

## Our Mission
To [mission statement] through cutting-edge technology and exceptional service.

## What We Do
- **Core Services**: [List main services]
- **Technology Stack**: [List technologies used]
- **Target Market**: [Describe target customers]

## Why Choose Us
- **Expertise**: [Years] of experience in [field]
- **Innovation**: [Specific innovations or achievements]
- **Support**: [Customer support details]

## Career Opportunities
We're always looking for talented individuals to join our team. Current openings include:
- [Job title 1]: [Brief description]
- [Job title 2]: [Brief description]

## Contact Information
- Website: [website]
- Email: [email]
- Phone: [phone]

---

Example 2 - Art Gallery Brochure:
# [Gallery Name] - Contemporary Art Gallery

## Welcome to [Gallery Name]
[Gallery Name] is a premier contemporary art gallery showcasing works by emerging and established artists.

## Our Collection
Our gallery features:
- **Contemporary Art**: Modern works by local and international artists
- **Exhibitions**: Regular rotating exhibitions
- **Artist Spotlights**: Featured artist programs

## Current Exhibition
**[Exhibition Name]**
[Brief description of current exhibition]

## Gallery Information
- **Location**: [Address]
- **Hours**: [Operating hours]
- **Admission**: [Pricing information]

## Visit Us
Experience the beauty of contemporary art at [Gallery Name].

---

Example 3 - Service Company Brochure:
# [Company Name] - Professional Services

## About [Company Name]
[Company Name] has been providing [services] for [number] years, serving clients across [geographic area].

## Our Services
- **[Service 1]**: [Description]
- **[Service 2]**: [Description]
- **[Service 3]**: [Description]

## Our Team
Our experienced professionals bring [X] years of combined experience to every project.

## Client Testimonials
"[Testimonial quote]" - [Client Name]

## Get Started
Contact us today to discuss your project needs.

---

Now create a brochure following these examples, adapting the format to the specific company and information provided."""
    
    def set_humorous_tone(self):
        """Set the system prompt to create humorous brochures."""
        self.brochure_system_prompt = """You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.

Here are examples of humorous brochure formats:

Example 1 - Tech Company (Humorous):
# [Company Name] - We're Not Just Another Tech Company! 🚀

## About Us (The Real Story)
[Company] is that tech company that actually makes software that works! Founded in [year], we've been confusing our competitors ever since.

## Our Mission (No, Really)
To make technology so easy that even your grandma can use it (and she does!).

## What We Actually Do
- **Core Services**: We build stuff that doesn't break (most of the time)
- **Technology Stack**: The latest and greatest (when it's not buggy)
- **Target Market**: Anyone brave enough to try new technology

## Why Choose Us (Seriously)
- **Expertise**: We've been doing this for [Years] years (and we're still learning!)
- **Innovation**: We invented the wheel... again... but this time it's digital!
- **Support**: Our support team actually responds (within 24-48 hours, we promise!)

## Career Opportunities (We're Hiring!)
Join our team of slightly crazy but brilliant people:
- **Coffee Drinker**: Must be able to consume 5+ cups daily
- **Problem Solver**: Should enjoy fixing things we broke yesterday
- **Team Player**: Must be okay with random nerf gun battles

## Contact Information
- Website: [website] (it actually works!)
- Email: [email] (we read these, really!)
- Phone: [phone] (if you prefer talking to humans)

---

Example 2 - Art Gallery (Humorous):
# [Gallery Name] - Where Art Meets Reality (Sort Of)

## Welcome to [Gallery Name]
[Gallery Name] is where we display art that makes you think "I could do that" (but you probably couldn't).

## Our Collection (The Good Stuff)
Our gallery features:
- **Contemporary Art**: Modern works that confuse and delight
- **Exhibitions**: We rotate them so often, even we forget what's up
- **Artist Spotlights**: Because every artist needs their 15 minutes

## Current Exhibition
**[Exhibition Name]**
[Brief description that sounds way more intellectual than it actually is]

## Gallery Information
- **Location**: [Address] (GPS recommended)
- **Hours**: When we feel like it (check our website)
- **Admission**: Free (but donations are appreciated... really appreciated)

## Visit Us
Come see art that will make you question everything (including your life choices).

---

Now create a humorous brochure following these examples, adapting the format to the specific company and information provided."""
    
    def demonstrate_multi_shot_prompting(self, company_name: str, url: str, company_type: str = "tech"):
        """
        Demonstrate multi-shot prompting with different company types.
        
        Args:
            company_name (str): Name of the company
            url (str): Company website URL
            company_type (str): Type of company ("tech", "gallery", "service")
            
        Returns:
            str: Generated brochure with multi-shot prompting
        """
        print(f"🎯 Demonstrating multi-shot prompting for {company_type} company...")
        
        # Set appropriate tone based on company type
        if company_type == "tech":
            self.set_professional_tone()
            print("📋 Using tech company examples for better link analysis")
        elif company_type == "gallery":
            self.set_professional_tone()
            print("🎨 Using art gallery examples for better link analysis")
        elif company_type == "service":
            self.set_professional_tone()
            print("💼 Using service company examples for better link analysis")
        
        # Generate brochure with multi-shot prompting
        brochure = self.create_brochure(company_name, url)
        return brochure


def main():
    """
    Example usage of the BrochureGenerator with multi-shot prompting.
    """
    print("🚀 Brochure Generator - AI-powered company brochure generation")
    print("🎯 Now with Multi-Shot Prompting for better results!")
    print("=" * 70)
    
    # Create generator
    generator = BrochureGenerator()
    
    # Example usage with multi-shot prompting
    company_name = "HuggingFace"
    url = "https://huggingface.co"
    company_type = "tech"
    
    print(f"📋 Generating brochure for {company_name}")
    print(f"🌐 Website: {url}")
    print(f"🏢 Company Type: {company_type}")
    print()
    
    try:
        # Demonstrate multi-shot prompting
        brochure = generator.demonstrate_multi_shot_prompting(company_name, url, company_type)
        print("✅ Brochure generated successfully with multi-shot prompting!")
        print()
        print("📄 Generated Brochure:")
        print("=" * 40)
        print(brochure)
        print("=" * 40)
        
        print("\n🎯 Multi-Shot Prompting Benefits:")
        print("• Better link analysis with multiple examples")
        print("• Improved brochure structure and formatting")
        print("• More consistent output quality")
        print("• Better understanding of different company types")
        
    except Exception as e:
        print(f"❌ Error generating brochure: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
