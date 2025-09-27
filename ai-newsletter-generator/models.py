"""
Data models and base classes for the AI Newsletter Generator
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class ContentSourceType(Enum):
    """Enumeration of content source types"""
    RSS = "rss"
    NEWS_API = "news_api"
    WEB_SCRAPER = "web_scraper"
    SOCIAL_MEDIA = "social_media"


class NewsletterFrequency(Enum):
    """Enumeration of newsletter frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ContentLength(Enum):
    """Enumeration of content length options"""
    BRIEF = "brief"
    MEDIUM = "medium"
    DETAILED = "detailed"


class WritingStyle(Enum):
    """Enumeration of writing styles"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    TECHNICAL = "technical"


@dataclass
class Article:
    """Represents a single article with all its metadata"""
    title: str
    description: str
    url: str
    source: str
    published_at: str
    content: str = ""
    image_url: str = ""
    summary: str = ""
    categories: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    quality_score: float = 0.0
    
    def __post_init__(self):
        """Validate article data after initialization"""
        if not self.title:
            raise ValueError("Article title cannot be empty")
        if not self.url:
            raise ValueError("Article URL cannot be empty")


@dataclass
class NewsletterConfig:
    """Configuration for newsletter generation"""
    user_interests: List[str]
    frequency: NewsletterFrequency
    content_length: ContentLength
    writing_style: WritingStyle
    language: str = "en"
    max_articles: int = 10
    include_images: bool = True
    include_analytics: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.user_interests:
            raise ValueError("User interests cannot be empty")
        if self.max_articles <= 0:
            raise ValueError("Max articles must be positive")


@dataclass
class Newsletter:
    """Represents a complete newsletter"""
    title: str
    content: str
    articles: List[Article]
    generated_at: datetime = field(default_factory=datetime.now)
    config: Optional[NewsletterConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_article(self, article: Article) -> None:
        """Add an article to the newsletter"""
        self.articles.append(article)
    
    def get_article_count(self) -> int:
        """Get the number of articles in the newsletter"""
        return len(self.articles)
    
    def get_topics(self) -> List[str]:
        """Get all unique topics from articles"""
        topics = set()
        for article in self.articles:
            topics.update(article.categories)
        return list(topics)


class ContentSource(ABC):
    """Abstract base class for content sources"""
    
    def __init__(self, name: str, source_type: ContentSourceType):
        self.name = name
        self.source_type = source_type
        self.is_active = True
    
    @abstractmethod
    def fetch_content(self, query: str, limit: int = 10) -> List[Article]:
        """Fetch content from the source"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the source is accessible"""
        pass
    
    def deactivate(self) -> None:
        """Deactivate the content source"""
        self.is_active = False
    
    def activate(self) -> None:
        """Activate the content source"""
        self.is_active = True


class AIService(ABC):
    """Abstract base class for AI services"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.is_configured = bool(api_key)
    
    @abstractmethod
    def summarize_content(self, content: str, max_length: int = 200) -> str:
        """Summarize content using AI"""
        pass
    
    @abstractmethod
    def classify_content(self, title: str, description: str) -> List[str]:
        """Classify content into categories"""
        pass
    
    @abstractmethod
    def generate_intro(self, topics: List[str], style: WritingStyle) -> str:
        """Generate newsletter introduction"""
        pass


class NewsletterTemplate(ABC):
    """Abstract base class for newsletter templates"""
    
    def __init__(self, name: str, style: WritingStyle):
        self.name = name
        self.style = style
    
    @abstractmethod
    def generate_html(self, newsletter: Newsletter) -> str:
        """Generate HTML for the newsletter"""
        pass
    
    @abstractmethod
    def get_css_styles(self) -> str:
        """Get CSS styles for the template"""
        pass


class AnalyticsService(ABC):
    """Abstract base class for analytics services"""
    
    @abstractmethod
    def track_newsletter_generation(self, newsletter: Newsletter) -> None:
        """Track newsletter generation"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass


class EmailService(ABC):
    """Abstract base class for email services"""
    
    @abstractmethod
    def send_newsletter(self, newsletter: Newsletter, recipients: List[str]) -> bool:
        """Send newsletter via email"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test email service connection"""
        pass
