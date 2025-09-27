#!/usr/bin/env python3
"""
Test suite for AI Newsletter Generator using Object-Oriented Programming
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NewsletterOOPTestSuite:
    """Test suite for OOP implementation"""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
    
    def run_test(self, test_name: str, test_func):
        """Run a single test"""
        print(f"\n🔍 Running {test_name}...")
        self.total_tests += 1
        
        try:
            if test_func():
                self.passed_tests += 1
                print(f"✅ {test_name} passed")
                return True
            else:
                print(f"❌ {test_name} failed")
                return False
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            return False
    
    def test_imports(self):
        """Test if all required packages can be imported"""
        print("🔍 Testing imports...")
        
        try:
            # Test basic imports
            import requests
            print("✅ requests imported successfully")
            
            import feedparser
            print("✅ feedparser imported successfully")
            
            from bs4 import BeautifulSoup
            print("✅ beautifulsoup4 imported successfully")
            
            import pandas as pd
            print("✅ pandas imported successfully")
            
            import plotly.express as px
            print("✅ plotly imported successfully")
            
            import streamlit as st
            print("✅ streamlit imported successfully")
            
            from openai import OpenAI
            print("✅ openai imported successfully")
            
            from dotenv import load_dotenv
            print("✅ python-dotenv imported successfully")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
    
    def test_models(self):
        """Test model classes"""
        try:
            from models import (
                Article, Newsletter, NewsletterConfig, ContentSource, AIService,
                WritingStyle, NewsletterFrequency, ContentLength, ContentSourceType
            )
            
            # Test Article creation
            article = Article(
                title="Test Article",
                description="Test description",
                url="https://example.com",
                source="Test Source",
                published_at="2024-01-01"
            )
            print("✅ Article model works")
            
            # Test NewsletterConfig
            config = NewsletterConfig(
                user_interests=["AI", "Technology"],
                frequency=NewsletterFrequency.WEEKLY,
                content_length=ContentLength.MEDIUM,
                writing_style=WritingStyle.PROFESSIONAL
            )
            print("✅ NewsletterConfig model works")
            
            # Test Newsletter
            newsletter = Newsletter(
                title="Test Newsletter",
                content="Test content",
                articles=[article],
                config=config
            )
            print("✅ Newsletter model works")
            
            return True
            
        except Exception as e:
            print(f"❌ Models test failed: {e}")
            return False
    
    def test_services(self):
        """Test service classes"""
        try:
            from services import (
                RSSSource, NewsAPISource, OpenAIService, 
                ProfessionalTemplate, CasualTemplate, SMTPEmailService, BasicAnalyticsService
            )
            
            # Test RSS source
            rss_source = RSSSource("https://feeds.feedburner.com/oreilly/radar")
            print("✅ RSSSource works")
            
            # Test NewsAPI source
            news_source = NewsAPISource("dummy-key")
            print("✅ NewsAPISource works")
            
            # Test AI service
            ai_service = OpenAIService("dummy-key")
            print("✅ OpenAIService works")
            
            # Test templates
            professional_template = ProfessionalTemplate()
            casual_template = CasualTemplate()
            print("✅ Templates work")
            
            # Test email service
            email_service = SMTPEmailService("smtp.gmail.com", 587, "user", "pass")
            print("✅ SMTPEmailService works")
            
            # Test analytics service
            analytics_service = BasicAnalyticsService()
            print("✅ BasicAnalyticsService works")
            
            return True
            
        except Exception as e:
            print(f"❌ Services test failed: {e}")
            return False
    
    def test_newsletter_generator(self):
        """Test newsletter generator"""
        try:
            from newsletter_generator import NewsletterGenerator, NewsletterManager
            from models import NewsletterConfig, WritingStyle, NewsletterFrequency, ContentLength
            
            # Test generator creation
            generator = NewsletterGenerator("dummy-key")
            print("✅ NewsletterGenerator works")
            
            # Test manager creation
            manager = NewsletterManager(generator)
            print("✅ NewsletterManager works")
            
            # Test configuration
            config = NewsletterConfig(
                user_interests=["AI", "Technology"],
                frequency=NewsletterFrequency.WEEKLY,
                content_length=ContentLength.MEDIUM,
                writing_style=WritingStyle.PROFESSIONAL
            )
            print("✅ Configuration works")
            
            return True
            
        except Exception as e:
            print(f"❌ Newsletter generator test failed: {e}")
            return False
    
    def test_content_sources(self):
        """Test content sources"""
        try:
            from services import RSSSource, NewsAPISource
            
            # Test RSS source
            rss_source = RSSSource("https://feeds.feedburner.com/oreilly/radar")
            print("✅ RSS source created")
            
            # Test NewsAPI source
            news_source = NewsAPISource("dummy-key")
            print("✅ NewsAPI source created")
            
            return True
            
        except Exception as e:
            print(f"❌ Content sources test failed: {e}")
            return False
    
    def test_ai_processing(self):
        """Test AI processing (without API calls)"""
        try:
            from services import OpenAIService
            
            # Test AI service creation
            ai_service = OpenAIService("dummy-key")
            print("✅ AI service created")
            
            # Test without API key (should handle gracefully)
            summary = ai_service.summarize_content("Test content", 100)
            print("✅ Summarization works (graceful degradation)")
            
            categories = ai_service.classify_content("Test title", "Test description")
            print("✅ Classification works (graceful degradation)")
            
            return True
            
        except Exception as e:
            print(f"❌ AI processing test failed: {e}")
            return False
    
    def test_templates(self):
        """Test newsletter templates"""
        try:
            from services import ProfessionalTemplate, CasualTemplate
            from models import Newsletter, Article, NewsletterConfig, WritingStyle, NewsletterFrequency, ContentLength
            
            # Create test data
            article = Article(
                title="Test Article",
                description="Test description",
                url="https://example.com",
                source="Test Source",
                published_at="2024-01-01",
                summary="Test summary",
                categories=["AI", "Technology"]
            )
            
            config = NewsletterConfig(
                user_interests=["AI", "Technology"],
                frequency=NewsletterFrequency.WEEKLY,
                content_length=ContentLength.MEDIUM,
                writing_style=WritingStyle.PROFESSIONAL
            )
            
            newsletter = Newsletter(
                title="Test Newsletter",
                content="",
                articles=[article],
                config=config
            )
            
            # Test professional template
            professional_template = ProfessionalTemplate()
            html = professional_template.generate_html(newsletter)
            print("✅ Professional template works")
            
            # Test casual template
            casual_template = CasualTemplate()
            html = casual_template.generate_html(newsletter)
            print("✅ Casual template works")
            
            return True
            
        except Exception as e:
            print(f"❌ Templates test failed: {e}")
            return False
    
    def test_analytics(self):
        """Test analytics functionality"""
        try:
            from services import BasicAnalyticsService
            from models import Newsletter, Article, NewsletterConfig, WritingStyle, NewsletterFrequency, ContentLength
            
            # Create analytics service
            analytics_service = BasicAnalyticsService()
            print("✅ Analytics service created")
            
            # Create test newsletter
            article = Article(
                title="Test Article",
                description="Test description",
                url="https://example.com",
                source="Test Source",
                published_at="2024-01-01"
            )
            
            config = NewsletterConfig(
                user_interests=["AI", "Technology"],
                frequency=NewsletterFrequency.WEEKLY,
                content_length=ContentLength.MEDIUM,
                writing_style=WritingStyle.PROFESSIONAL
            )
            
            newsletter = Newsletter(
                title="Test Newsletter",
                content="Test content",
                articles=[article],
                config=config
            )
            
            # Test tracking
            analytics_service.track_newsletter_generation(newsletter)
            print("✅ Analytics tracking works")
            
            # Test metrics
            metrics = analytics_service.get_performance_metrics()
            print("✅ Analytics metrics work")
            
            return True
            
        except Exception as e:
            print(f"❌ Analytics test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 AI Newsletter Generator - Test Suite")
        print("=" * 50)
        
        tests = [
            ("Import Test", self.test_imports),
            ("Models Test", self.test_models),
            ("Services Test", self.test_services),
            ("Newsletter Generator Test", self.test_newsletter_generator),
            ("Content Sources Test", self.test_content_sources),
            ("AI Processing Test", self.test_ai_processing),
            ("Templates Test", self.test_templates),
            ("Analytics Test", self.test_analytics)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print results
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.passed_tests == self.total_tests:
            print("🎉 All tests passed! The newsletter generator is ready to use.")
            print("\n📝 Next steps:")
            print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
            print("2. Run the demo: python demo.py")
            print("3. Start the web interface: streamlit run app.py")
        else:
            print("❌ Some tests failed. Please check the output above for details.")
            print("\n🔧 To fix issues:")
            print("1. Install missing packages: pip install -r requirements.txt")
            print("2. Check your Python environment")
            print("3. Ensure you have internet access for RSS feeds")
        
        return self.passed_tests == self.total_tests


def main():
    """Main function"""
    test_suite = NewsletterOOPTestSuite()
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
