"""
Streamlit application for AI Newsletter Generator using Object-Oriented Programming
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

from newsletter_generator import NewsletterGenerator, NewsletterManager
from models import NewsletterConfig, WritingStyle, NewsletterFrequency, ContentLength


class NewsletterApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.generator = None
        self.manager = None
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Newsletter Generator",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 AI Newsletter Generator")
        st.markdown("Create personalized newsletters using AI-powered content curation")
    
    def initialize_services(self, openai_key: str, news_api_key: str = None):
        """Initialize the newsletter generator and manager"""
        if not self.generator:
            self.generator = NewsletterGenerator(
                openai_api_key=openai_key,
                news_api_key=news_api_key
            )
            self.manager = NewsletterManager(self.generator)
    
    def render_sidebar(self):
        """Render the sidebar configuration"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # User interests
            st.subheader("Your Interests")
            available_interests = [
                "Technology", "AI", "Machine Learning", "Data Science", "Programming",
                "Startups", "Business", "Science", "Health", "Environment", 
                "Politics", "Sports", "Entertainment", "Finance", "Education"
            ]
            
            interests = st.multiselect(
                "Select topics you're interested in:",
                available_interests,
                default=["Technology", "AI", "Machine Learning"]
            )
            
            # Newsletter settings
            st.subheader("Newsletter Settings")
            frequency = st.selectbox(
                "Frequency:",
                [freq.value for freq in NewsletterFrequency],
                index=1  # Default to weekly
            )
            
            content_length = st.selectbox(
                "Content Length:",
                [length.value for length in ContentLength],
                index=1  # Default to medium
            )
            
            writing_style = st.selectbox(
                "Writing Style:",
                [style.value for style in WritingStyle],
                index=0  # Default to professional
            )
            
            max_articles = st.slider("Max Articles:", 5, 20, 10)
            
            # API Keys
            st.subheader("API Configuration")
            openai_key = st.text_input("OpenAI API Key:", type="password")
            news_api_key = st.text_input("News API Key (optional):", type="password")
            
            return {
                "interests": interests,
                "frequency": frequency,
                "content_length": content_length,
                "writing_style": writing_style,
                "max_articles": max_articles,
                "openai_key": openai_key,
                "news_api_key": news_api_key
            }
    
    def render_main_content(self, config_data: dict):
        """Render the main content area"""
        if st.button("🚀 Generate Newsletter", type="primary"):
            if not config_data["openai_key"]:
                st.error("Please enter your OpenAI API key")
                return
            
            # Initialize services
            self.initialize_services(
                config_data["openai_key"],
                config_data["news_api_key"]
            )
            
            # Create configuration
            config = NewsletterConfig(
                user_interests=config_data["interests"],
                frequency=NewsletterFrequency(config_data["frequency"]),
                content_length=ContentLength(config_data["content_length"]),
                writing_style=WritingStyle(config_data["writing_style"]),
                max_articles=config_data["max_articles"]
            )
            
            # Generate newsletter
            with st.spinner("Generating your personalized newsletter..."):
                try:
                    # Get preview first
                    preview = self.generator.get_newsletter_preview(config)
                    
                    # Show preview
                    with st.expander("📊 Newsletter Preview", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Articles Found", preview["total_articles_found"])
                        with col2:
                            st.metric("Relevant Articles", preview["relevant_articles"])
                        with col3:
                            st.metric("Reading Time", f"{preview['estimated_reading_time']} min")
                        
                        st.write("**Topics:**", ", ".join(preview["topics"]))
                        st.write("**Sources:**", ", ".join(preview["sources"]))
                    
                    # Generate newsletter
                    newsletter = self.manager.create_newsletter(config)
                    
                    # Display newsletter
                    st.markdown("## 📰 Your Personalized Newsletter")
                    st.components.v1.html(newsletter.content, height=800, scrolling=True)
                    
                    # Download option
                    st.download_button(
                        "📥 Download Newsletter",
                        newsletter.content,
                        file_name=f"newsletter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                    
                    # Show article details
                    with st.expander("📋 Article Details"):
                        for i, article in enumerate(newsletter.articles, 1):
                            st.write(f"**{i}. {article.title}**")
                            st.write(f"*Source: {article.source} | Relevance: {article.relevance_score:.2f}*")
                            st.write(f"*Categories: {', '.join(article.categories)}*")
                            st.write(f"*Summary: {article.summary}*")
                            st.write(f"[Read more]({article.url})")
                            st.divider()
                    
                except Exception as e:
                    st.error(f"Error generating newsletter: {str(e)}")
                    st.write("Please check your API keys and try again.")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.markdown("---")
        st.header("📊 Newsletter Analytics")
        
        if self.manager:
            stats = self.manager.get_newsletter_stats()
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Newsletters", stats.get("total_newsletters", 0))
            
            with col2:
                st.metric("Total Articles", stats.get("total_articles", 0))
            
            with col3:
                avg_articles = stats.get("average_articles_per_newsletter", 0)
                st.metric("Avg Articles/Newsletter", f"{avg_articles:.1f}")
            
            with col4:
                if stats.get("latest_newsletter_date"):
                    st.metric("Latest Newsletter", stats["latest_newsletter_date"].strftime("%Y-%m-%d"))
            
            # Analytics data
            analytics = stats.get("analytics", {})
            if analytics:
                st.subheader("Generation Analytics")
                analytics_df = pd.DataFrame([analytics])
                st.dataframe(analytics_df, use_container_width=True)
        else:
            st.info("Generate your first newsletter to see analytics!")
    
    def render_content_sources(self):
        """Render content sources management"""
        st.markdown("---")
        st.header("📡 Content Sources")
        
        if self.generator:
            # Test sources
            if st.button("🔍 Test All Sources"):
                with st.spinner("Testing content sources..."):
                    results = self.generator.test_all_sources()
                    
                    for source_name, is_working in results.items():
                        if is_working:
                            st.success(f"✅ {source_name} - Working")
                        else:
                            st.error(f"❌ {source_name} - Not accessible")
            
            # Show active sources
            active_sources = self.generator.get_active_sources()
            st.subheader("Active Sources")
            for source in active_sources:
                st.write(f"• {source.name} ({source.source_type.value})")
        else:
            st.info("Initialize the generator to manage content sources")
    
    def render_newsletter_history(self):
        """Render newsletter history"""
        st.markdown("---")
        st.header("📚 Newsletter History")
        
        if self.manager:
            history = self.manager.get_newsletter_history()
            
            if history:
                for i, newsletter in enumerate(reversed(history[-5:]), 1):  # Show last 5
                    with st.expander(f"Newsletter {len(history) - i + 1} - {newsletter.generated_at.strftime('%Y-%m-%d %H:%M')}"):
                        st.write(f"**Articles:** {len(newsletter.articles)}")
                        st.write(f"**Topics:** {', '.join(newsletter.get_topics())}")
                        st.write(f"**Style:** {newsletter.config.writing_style.value if newsletter.config else 'Unknown'}")
                        
                        if st.button(f"View Newsletter {len(history) - i + 1}", key=f"view_{i}"):
                            st.components.v1.html(newsletter.content, height=600, scrolling=True)
            else:
                st.info("No newsletter history yet. Generate your first newsletter!")
        else:
            st.info("Initialize the generator to view newsletter history")
    
    def run(self):
        """Run the Streamlit application"""
        # Render sidebar and get configuration
        config_data = self.render_sidebar()
        
        # Render main content
        self.render_main_content(config_data)
        
        # Render analytics
        self.render_analytics()
        
        # Render content sources
        self.render_content_sources()
        
        # Render newsletter history
        self.render_newsletter_history()


def main():
    """Main function to run the application"""
    app = NewsletterApp()
    app.run()


if __name__ == "__main__":
    main()
