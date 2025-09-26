"""
Main Newsletter Generator class using Object-Oriented Programming
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from models import (
    Newsletter, NewsletterConfig, Article, ContentSource, AIService, 
    NewsletterTemplate, AnalyticsService, EmailService, WritingStyle
)
from services import (
    NewsAPISource, RSSSource, OpenAIService, ProfessionalTemplate, 
    CasualTemplate, SMTPEmailService, BasicAnalyticsService
)


class NewsletterGenerator:
    """
    Main newsletter generator class that orchestrates all components
    using proper Object-Oriented Programming principles
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 news_api_key: Optional[str] = None,
                 email_service: Optional[EmailService] = None,
                 analytics_service: Optional[AnalyticsService] = None):
        """
        Initialize the newsletter generator with all required services
        
        Args:
            openai_api_key: OpenAI API key for AI processing
            news_api_key: NewsAPI key for news aggregation
            email_service: Email service for sending newsletters
            analytics_service: Analytics service for tracking
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize AI service
        self.ai_service = OpenAIService(
            openai_api_key or os.getenv("OPENAI_API_KEY", "")
        )
        
        # Initialize content sources
        self.content_sources: List[ContentSource] = []
        self._setup_content_sources(news_api_key)
        
        # Initialize templates
        self.templates: Dict[WritingStyle, NewsletterTemplate] = {
            WritingStyle.PROFESSIONAL: ProfessionalTemplate(),
            WritingStyle.CASUAL: CasualTemplate()
        }
        
        # Initialize services
        self.email_service = email_service
        self.analytics_service = analytics_service or BasicAnalyticsService()
        
        # Newsletter generation settings
        self.max_articles_per_source = 5
        self.min_relevance_score = 0.3
    
    def _setup_content_sources(self, news_api_key: Optional[str] = None) -> None:
        """Setup content sources"""
        # Add NewsAPI if key is provided
        if news_api_key or os.getenv("NEWS_API_KEY"):
            api_key = news_api_key or os.getenv("NEWS_API_KEY")
            self.content_sources.append(NewsAPISource(api_key))
        
        # Add RSS feeds
        rss_feeds = [
            "https://feeds.feedburner.com/oreilly/radar",
            "https://techcrunch.com/feed/",
            "https://feeds.feedburner.com/venturebeat/SZYF",
            "https://www.wired.com/feed/rss",
            "https://feeds.feedburner.com/arstechnica/index/",
            "https://feeds.feedburner.com/oreilly/radar"
        ]
        
        for feed_url in rss_feeds:
            self.content_sources.append(RSSSource(feed_url))
    
    def add_content_source(self, source: ContentSource) -> None:
        """Add a new content source"""
        self.content_sources.append(source)
    
    def remove_content_source(self, source_name: str) -> bool:
        """Remove a content source by name"""
        for source in self.content_sources:
            if source.name == source_name:
                self.content_sources.remove(source)
                return True
        return False
    
    def get_active_sources(self) -> List[ContentSource]:
        """Get all active content sources"""
        return [source for source in self.content_sources if source.is_active]
    
    def test_all_sources(self) -> Dict[str, bool]:
        """Test all content sources and return their status"""
        results = {}
        for source in self.content_sources:
            results[source.name] = source.test_connection()
        return results
    
    def fetch_content(self, interests: List[str]) -> List[Article]:
        """Fetch content from all active sources"""
        all_articles = []
        
        for source in self.get_active_sources():
            for interest in interests:
                try:
                    articles = source.fetch_content(interest, self.max_articles_per_source)
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"Error fetching from {source.name}: {e}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles
    
    def process_articles(self, articles: List[Article], config: NewsletterConfig) -> List[Article]:
        """Process articles with AI services"""
        processed_articles = []
        
        for article in articles:
            # Summarize content based on length preference
            max_lengths = {
                config.content_length.BRIEF: 100,
                config.content_length.MEDIUM: 200,
                config.content_length.DETAILED: 300
            }
            max_length = max_lengths.get(config.content_length, 200)
            
            # Summarize content
            article.summary = self.ai_service.summarize_content(
                article.description or article.content,
                max_length
            )
            
            # Classify content
            article.categories = self.ai_service.classify_content(
                article.title,
                article.description
            )
            
            # Calculate relevance score
            article.relevance_score = self._calculate_relevance_score(
                article.categories, config.user_interests
            )
            
            # Filter by relevance
            if article.relevance_score >= self.min_relevance_score:
                processed_articles.append(article)
        
        # Sort by relevance and recency
        processed_articles.sort(
            key=lambda x: (x.relevance_score, x.published_at), 
            reverse=True
        )
        
        return processed_articles[:config.max_articles]
    
    def _calculate_relevance_score(self, categories: List[str], interests: List[str]) -> float:
        """Calculate relevance score for an article"""
        if not categories or not interests:
            return 0.0
        
        # Simple scoring: count matching categories
        matches = sum(1 for cat in categories if any(interest.lower() in cat.lower() for interest in interests))
        return matches / len(interests) if interests else 0.0
    
    def generate_newsletter(self, config: NewsletterConfig) -> Newsletter:
        """Generate a complete newsletter"""
        # Fetch content
        articles = self.fetch_content(config.user_interests)
        
        if not articles:
            raise ValueError("No content found for the specified interests")
        
        # Process articles
        processed_articles = self.process_articles(articles, config)
        
        if not processed_articles:
            raise ValueError("No relevant content found after processing")
        
        # Create newsletter
        newsletter = Newsletter(
            title=f"AI Newsletter - {datetime.now().strftime('%B %d, %Y')}",
            content="",  # Will be filled by template
            articles=processed_articles,
            config=config
        )
        
        # Generate content using template
        template = self.templates.get(config.writing_style, self.templates[WritingStyle.PROFESSIONAL])
        newsletter.content = template.generate_html(newsletter)
        
        # Track analytics
        self.analytics_service.track_newsletter_generation(newsletter)
        
        return newsletter
    
    def send_newsletter(self, newsletter: Newsletter, recipients: List[str]) -> bool:
        """Send newsletter via email"""
        if not self.email_service:
            print("No email service configured")
            return False
        
        return self.email_service.send_newsletter(newsletter, recipients)
    
    def get_analytics(self) -> Dict[str, any]:
        """Get analytics data"""
        return self.analytics_service.get_performance_metrics()
    
    def configure_email_service(self, smtp_server: str, smtp_port: int, 
                               username: str, password: str) -> None:
        """Configure email service"""
        self.email_service = SMTPEmailService(smtp_server, smtp_port, username, password)
    
    def get_newsletter_preview(self, config: NewsletterConfig) -> Dict[str, any]:
        """Get a preview of what the newsletter will contain"""
        articles = self.fetch_content(config.user_interests)
        processed_articles = self.process_articles(articles, config)
        
        return {
            "total_articles_found": len(articles),
            "relevant_articles": len(processed_articles),
            "topics": list(set([cat for article in processed_articles for cat in article.categories])),
            "sources": list(set([article.source for article in processed_articles])),
            "estimated_reading_time": len(processed_articles) * 2  # 2 minutes per article
        }


class NewsletterManager:
    """
    High-level manager for newsletter operations
    """
    
    def __init__(self, generator: NewsletterGenerator):
        self.generator = generator
        self.newsletter_history: List[Newsletter] = []
    
    def create_newsletter(self, config: NewsletterConfig) -> Newsletter:
        """Create a new newsletter"""
        newsletter = self.generator.generate_newsletter(config)
        self.newsletter_history.append(newsletter)
        return newsletter
    
    def get_newsletter_history(self) -> List[Newsletter]:
        """Get newsletter history"""
        return self.newsletter_history.copy()
    
    def get_latest_newsletter(self) -> Optional[Newsletter]:
        """Get the latest newsletter"""
        return self.newsletter_history[-1] if self.newsletter_history else None
    
    def save_newsletter(self, newsletter: Newsletter, filename: str) -> bool:
        """Save newsletter to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(newsletter.content)
            return True
        except Exception as e:
            print(f"Error saving newsletter: {e}")
            return False
    
    def get_newsletter_stats(self) -> Dict[str, any]:
        """Get newsletter statistics"""
        if not self.newsletter_history:
            return {"total_newsletters": 0}
        
        total_articles = sum(len(newsletter.articles) for newsletter in self.newsletter_history)
        
        return {
            "total_newsletters": len(self.newsletter_history),
            "total_articles": total_articles,
            "average_articles_per_newsletter": total_articles / len(self.newsletter_history),
            "latest_newsletter_date": self.newsletter_history[-1].generated_at,
            "analytics": self.generator.get_analytics()
        }
