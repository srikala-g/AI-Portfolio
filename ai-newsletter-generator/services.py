"""
Concrete implementations of services for the AI Newsletter Generator
"""

import os
import requests
import feedparser
from typing import List, Dict, Optional
from datetime import datetime
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from models import (
    ContentSource, AIService, NewsletterTemplate, AnalyticsService, EmailService,
    Article, Newsletter, ContentSourceType, WritingStyle
)


class NewsAPISource(ContentSource):
    """NewsAPI content source implementation"""
    
    def __init__(self, api_key: str):
        super().__init__("NewsAPI", ContentSourceType.NEWS_API)
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_content(self, query: str, limit: int = 10) -> List[Article]:
        """Fetch news articles from NewsAPI"""
        if not self.is_active:
            return []
        
        try:
            url = f"{self.base_url}/everything"
            params = {
                "q": query,
                "apiKey": self.api_key,
                "pageSize": limit,
                "sortBy": "publishedAt",
                "language": "en"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            articles_data = response.json().get("articles", [])
            articles = []
            
            for article_data in articles_data:
                article = Article(
                    title=article_data.get("title", ""),
                    description=article_data.get("description", ""),
                    url=article_data.get("url", ""),
                    source=article_data.get("source", {}).get("name", ""),
                    published_at=article_data.get("publishedAt", ""),
                    content=article_data.get("content", ""),
                    image_url=article_data.get("urlToImage", "")
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test NewsAPI connection"""
        try:
            url = f"{self.base_url}/sources"
            params = {"apiKey": self.api_key}
            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except:
            return False


class RSSSource(ContentSource):
    """RSS feed content source implementation"""
    
    def __init__(self, feed_url: str):
        super().__init__("RSS", ContentSourceType.RSS)
        self.feed_url = feed_url
    
    def fetch_content(self, query: str, limit: int = 10) -> List[Article]:
        """Fetch content from RSS feed"""
        if not self.is_active:
            return []
        
        try:
            feed = feedparser.parse(self.feed_url)
            articles = []
            
            for entry in feed.entries[:limit]:
                article = Article(
                    title=entry.get("title", ""),
                    description=entry.get("description", ""),
                    url=entry.get("link", ""),
                    source=feed.feed.get("title", "RSS Feed"),
                    published_at=entry.get("published", ""),
                    content=entry.get("summary", ""),
                    image_url=""
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            print(f"Error fetching from RSS: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Test RSS feed connection"""
        try:
            feed = feedparser.parse(self.feed_url)
            return len(feed.entries) > 0
        except:
            return False


class OpenAIService(AIService):
    """OpenAI service implementation"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        if self.is_configured:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def summarize_content(self, content: str, max_length: int = 200) -> str:
        """Summarize content using OpenAI"""
        if not self.is_configured or not self.client:
            return content[:max_length] + "..."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Summarize the following content in {max_length} characters or less:"},
                    {"role": "user", "content": content}
                ],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing content: {e}")
            return content[:max_length] + "..."
    
    def classify_content(self, title: str, description: str) -> List[str]:
        """Classify content into categories"""
        if not self.is_configured or not self.client:
            return ["General"]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Classify the following content into relevant categories. Return only the category names separated by commas:"},
                    {"role": "user", "content": f"Title: {title}\nDescription: {description}"}
                ],
                max_tokens=50
            )
            categories = response.choices[0].message.content.strip().split(",")
            return [cat.strip() for cat in categories]
        except Exception as e:
            print(f"Error classifying content: {e}")
            return ["General"]
    
    def generate_intro(self, topics: List[str], style: WritingStyle) -> str:
        """Generate newsletter introduction"""
        if not self.is_configured or not self.client:
            return f"Welcome to your personalized newsletter covering {', '.join(topics)}!"
        
        try:
            style_prompts = {
                WritingStyle.PROFESSIONAL: "Write a professional newsletter introduction",
                WritingStyle.CASUAL: "Write a casual, friendly newsletter introduction",
                WritingStyle.TECHNICAL: "Write a technical newsletter introduction"
            }
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{style_prompts.get(style, style_prompts[WritingStyle.PROFESSIONAL])} for topics: {', '.join(topics)}"},
                    {"role": "user", "content": f"Topics: {', '.join(topics)}"}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating intro: {e}")
            return f"Welcome to your personalized newsletter covering {', '.join(topics)}!"


class ProfessionalTemplate(NewsletterTemplate):
    """Professional newsletter template"""
    
    def __init__(self):
        super().__init__("Professional", WritingStyle.PROFESSIONAL)
    
    def generate_html(self, newsletter: Newsletter) -> str:
        """Generate professional HTML newsletter"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{newsletter.title}</title>
            <style>{self.get_css_styles()}</style>
        </head>
        <body>
            <div class="header">
                <h1>📰 {newsletter.title}</h1>
                <p class="date">{newsletter.generated_at.strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="content">
        """
        
        for i, article in enumerate(newsletter.articles, 1):
            categories_html = "".join([f'<span class="category">{cat}</span>' for cat in article.categories])
            
            html += f"""
                <div class="article">
                    <h3>{i}. {article.title}</h3>
                    <div class="article-meta">
                        <span class="source">{article.source}</span> | 
                        <span class="date">{article.published_at}</span>
                    </div>
                    <p class="summary">{article.summary or article.description}</p>
                    <div class="categories">{categories_html}</div>
                    <p class="read-more"><a href="{article.url}" target="_blank">Read more →</a></p>
                </div>
            """
        
        html += """
            </div>
            
            <div class="footer">
                <p>Generated by AI Newsletter Generator | Powered by OpenAI</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_css_styles(self) -> str:
        """Get professional CSS styles"""
        return """
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #ffffff;
            }
            .header { 
                text-align: center; 
                border-bottom: 3px solid #007bff; 
                padding-bottom: 20px; 
                margin-bottom: 30px; 
            }
            .header h1 { color: #007bff; margin-bottom: 10px; }
            .date { color: #666; font-size: 0.9em; }
            .article { 
                margin-bottom: 30px; 
                padding: 20px; 
                border: 1px solid #e9ecef; 
                border-radius: 8px; 
                background-color: #f8f9fa;
            }
            .article h3 { color: #007bff; margin-top: 0; }
            .article-meta { color: #666; font-size: 0.9em; margin-bottom: 10px; }
            .category { 
                display: inline-block; 
                background-color: #007bff; 
                color: white; 
                padding: 2px 8px; 
                border-radius: 3px; 
                font-size: 0.8em; 
                margin-right: 5px; 
            }
            .read-more a { color: #007bff; text-decoration: none; font-weight: bold; }
            .read-more a:hover { text-decoration: underline; }
            .footer { 
                text-align: center; 
                margin-top: 40px; 
                padding-top: 20px; 
                border-top: 1px solid #ddd; 
                color: #666; 
            }
        """


class CasualTemplate(NewsletterTemplate):
    """Casual newsletter template"""
    
    def __init__(self):
        super().__init__("Casual", WritingStyle.CASUAL)
    
    def generate_html(self, newsletter: Newsletter) -> str:
        """Generate casual HTML newsletter"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{newsletter.title}</title>
            <style>{self.get_css_styles()}</style>
        </head>
        <body>
            <div class="header">
                <h1>🎉 {newsletter.title}</h1>
                <p class="date">{newsletter.generated_at.strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="content">
        """
        
        for i, article in enumerate(newsletter.articles, 1):
            categories_html = "".join([f'<span class="category">{cat}</span>' for cat in article.categories])
            
            html += f"""
                <div class="article">
                    <h3>{i}. {article.title}</h3>
                    <div class="article-meta">
                        <span class="source">📰 {article.source}</span> | 
                        <span class="date">📅 {article.published_at}</span>
                    </div>
                    <p class="summary">{article.summary or article.description}</p>
                    <div class="categories">{categories_html}</div>
                    <p class="read-more"><a href="{article.url}" target="_blank">👉 Read more</a></p>
                </div>
            """
        
        html += """
            </div>
            
            <div class="footer">
                <p>😊 Generated by AI Newsletter Generator | Made with ❤️</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def get_css_styles(self) -> str:
        """Get casual CSS styles"""
        return """
            body { 
                font-family: 'Comic Sans MS', cursive, sans-serif; 
                line-height: 1.6; 
                color: #333; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                background-color: #f0f8ff;
            }
            .header { 
                text-align: center; 
                border-bottom: 3px solid #28a745; 
                padding-bottom: 20px; 
                margin-bottom: 30px; 
            }
            .header h1 { color: #28a745; margin-bottom: 10px; }
            .date { color: #666; font-size: 0.9em; }
            .article { 
                margin-bottom: 30px; 
                padding: 20px; 
                border: 2px solid #28a745; 
                border-radius: 15px; 
                background-color: #ffffff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .article h3 { color: #28a745; margin-top: 0; }
            .article-meta { color: #666; font-size: 0.9em; margin-bottom: 10px; }
            .category { 
                display: inline-block; 
                background-color: #28a745; 
                color: white; 
                padding: 3px 10px; 
                border-radius: 15px; 
                font-size: 0.8em; 
                margin-right: 5px; 
            }
            .read-more a { color: #28a745; text-decoration: none; font-weight: bold; }
            .read-more a:hover { text-decoration: underline; }
            .footer { 
                text-align: center; 
                margin-top: 40px; 
                padding-top: 20px; 
                border-top: 1px solid #ddd; 
                color: #666; 
            }
        """


class SMTPEmailService(EmailService):
    """SMTP email service implementation"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_newsletter(self, newsletter: Newsletter, recipients: List[str]) -> bool:
        """Send newsletter via SMTP"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = newsletter.title
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            
            html_content = newsletter.content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test SMTP connection"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
            return True
        except:
            return False


class BasicAnalyticsService(AnalyticsService):
    """Basic analytics service implementation"""
    
    def __init__(self):
        self.metrics = {
            "newsletters_generated": 0,
            "total_articles": 0,
            "average_articles_per_newsletter": 0
        }
    
    def track_newsletter_generation(self, newsletter: Newsletter) -> None:
        """Track newsletter generation"""
        self.metrics["newsletters_generated"] += 1
        self.metrics["total_articles"] += len(newsletter.articles)
        self.metrics["average_articles_per_newsletter"] = (
            self.metrics["total_articles"] / self.metrics["newsletters_generated"]
        )
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """Get performance metrics"""
        return self.metrics.copy()
