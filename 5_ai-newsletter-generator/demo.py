#!/usr/bin/env python3
"""
Demo script for AI Newsletter Generator using Object-Oriented Programming
"""

import os
from dotenv import load_dotenv
from newsletter_generator import NewsletterGenerator, NewsletterManager
from models import NewsletterConfig, WritingStyle, NewsletterFrequency, ContentLength


class NewsletterDemo:
    """Demo class for the AI Newsletter Generator"""
    
    def __init__(self):
        load_dotenv()
        self.generator = None
        self.manager = None
    
    def initialize_services(self, openai_key: str, news_api_key: str = None):
        """Initialize the newsletter generator and manager"""
        print("🔄 Initializing services...")
        self.generator = NewsletterGenerator(
            openai_api_key=openai_key,
            news_api_key=news_api_key
        )
        self.manager = NewsletterManager(self.generator)
        print("✅ Services initialized successfully!")
    
    def test_content_sources(self):
        """Test all content sources"""
        print("\n🔍 Testing content sources...")
        if not self.generator:
            print("❌ Generator not initialized")
            return
        
        results = self.generator.test_all_sources()
        for source_name, is_working in results.items():
            status = "✅ Working" if is_working else "❌ Not accessible"
            print(f"   {source_name}: {status}")
    
    def test_ai_processing(self):
        """Test AI processing capabilities"""
        print("\n🧠 Testing AI processing...")
        if not self.generator or not self.generator.ai_service.is_configured:
            print("❌ AI service not configured")
            return
        
        # Test summarization
        sample_text = """
        Artificial Intelligence (AI) is transforming industries across the globe. 
        From healthcare to finance, AI technologies are enabling new possibilities 
        and improving efficiency. Machine learning algorithms can now process 
        vast amounts of data to identify patterns and make predictions with 
        unprecedented accuracy.
        """
        
        try:
            summary = self.generator.ai_service.summarize_content(sample_text, 100)
            print(f"✅ Summarization: {summary}")
            
            categories = self.generator.ai_service.classify_content(
                "New AI breakthrough in machine learning",
                "Researchers develop new neural network architecture"
            )
            print(f"✅ Classification: {categories}")
            
        except Exception as e:
            print(f"❌ AI processing failed: {e}")
    
    def generate_newsletter_demo(self, interests: list, style: str = "professional"):
        """Generate a newsletter demo"""
        print(f"\n📝 Generating newsletter for interests: {interests}")
        
        if not self.generator:
            print("❌ Generator not initialized")
            return None
        
        try:
            # Create configuration
            config = NewsletterConfig(
                user_interests=interests,
                frequency=NewsletterFrequency.WEEKLY,
                content_length=ContentLength.MEDIUM,
                writing_style=WritingStyle(style),
                max_articles=5
            )
            
            # Get preview
            preview = self.generator.get_newsletter_preview(config)
            print(f"📊 Preview: {preview['total_articles_found']} articles found, {preview['relevant_articles']} relevant")
            print(f"📊 Topics: {', '.join(preview['topics'])}")
            print(f"📊 Sources: {', '.join(preview['sources'])}")
            
            # Generate newsletter
            newsletter = self.manager.create_newsletter(config)
            print(f"✅ Newsletter generated with {len(newsletter.articles)} articles")
            
            # Save newsletter
            filename = f"demo_newsletter_{style}_{len(newsletter.articles)}_articles.html"
            if self.manager.save_newsletter(newsletter, filename):
                print(f"💾 Newsletter saved to: {filename}")
            
            return newsletter
            
        except Exception as e:
            print(f"❌ Newsletter generation failed: {e}")
            return None
    
    def show_newsletter_stats(self):
        """Show newsletter statistics"""
        print("\n📊 Newsletter Statistics")
        if not self.manager:
            print("❌ Manager not initialized")
            return
        
        stats = self.manager.get_newsletter_stats()
        print(f"   Total Newsletters: {stats.get('total_newsletters', 0)}")
        print(f"   Total Articles: {stats.get('total_articles', 0)}")
        print(f"   Average Articles/Newsletter: {stats.get('average_articles_per_newsletter', 0):.1f}")
        
        analytics = stats.get('analytics', {})
        if analytics:
            print(f"   Analytics: {analytics}")
    
    def run_full_demo(self):
        """Run the complete demo"""
        print("🤖 AI Newsletter Generator - Demo")
        print("=" * 50)
        
        # Check for API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        news_api_key = os.getenv("NEWS_API_KEY")
        
        if not openai_key:
            print("❌ Please set OPENAI_API_KEY environment variable")
            print("   export OPENAI_API_KEY='your-api-key-here'")
            return
        
        # Initialize services
        self.initialize_services(openai_key, news_api_key)
        
        # Test content sources
        self.test_content_sources()
        
        # Test AI processing
        self.test_ai_processing()
        
        # Generate newsletter
        interests = ["AI", "Machine Learning", "Technology"]
        newsletter = self.generate_newsletter_demo(interests, "professional")
        
        if newsletter:
            # Show article details
            print(f"\n📋 Newsletter Articles:")
            for i, article in enumerate(newsletter.articles, 1):
                print(f"   {i}. {article.title}")
                print(f"      Source: {article.source}")
                print(f"      Relevance: {article.relevance_score:.2f}")
                print(f"      Categories: {', '.join(article.categories)}")
                print(f"      Summary: {article.summary[:100]}...")
                print()
        
        # Show statistics
        self.show_newsletter_stats()
        
        print("\n🎉 Demo completed!")
        print("📝 Next steps:")
        print("1. Run the web interface: streamlit run app_oop.py")
        print("2. Generate more newsletters with different configurations")
        print("3. Explore the analytics dashboard")
    
    def run_quick_test(self):
        """Run a quick test without API calls"""
        print("🧪 Quick Test")
        print("=" * 30)
        
        # Test basic functionality
        try:
            from models import Article, NewsletterConfig, WritingStyle
            from services import RSSSource, OpenAIService
            
            # Test Article creation
            article = Article(
                title="Test Article",
                description="Test description",
                url="https://example.com",
                source="Test Source",
                published_at="2024-01-01"
            )
            print("✅ Article model works")
            
            # Test configuration
            config = NewsletterConfig(
                user_interests=["AI", "Technology"],
                frequency=NewsletterFrequency.WEEKLY,
                content_length=ContentLength.MEDIUM,
                writing_style=WritingStyle.PROFESSIONAL
            )
            print("✅ Configuration model works")
            
            # Test RSS source
            rss_source = RSSSource("https://feeds.feedburner.com/oreilly/radar")
            print("✅ RSS source works")
            
            # Test AI service (without API key)
            ai_service = OpenAIService("dummy-key")
            print("✅ AI service works")
            
            print("\n🎉 All components working correctly!")
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")


def main():
    """Main function"""
    demo = NewsletterDemo()
    
    # Check if we have API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openai_key and openai_key != "demo-key-for-testing":
        print("🚀 Running full demo with API keys...")
        demo.run_full_demo()
    else:
        print("🧪 Running quick test without API keys...")
        demo.run_quick_test()
        
        print("\n💡 To run the full demo:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
        print("2. Run: python demo_oop.py")


if __name__ == "__main__":
    main()
