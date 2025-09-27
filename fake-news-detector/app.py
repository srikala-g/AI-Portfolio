"""
Streamlit application for Fake News Detector using Object-Oriented Programming
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from fake_news_detector import FakeNewsDetector, DetectionPipeline, ModelManager
from models import Article, ModelType, DetectionStatus, ConfidenceLevel
from services import (
    AdvancedTextProcessor, LinguisticFeatureExtractor, SemanticFeatureExtractor,
    MLClassifier, SourceCredibilityChecker, APIFactChecker
)


class FakeNewsApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.detector = None
        self.pipeline = None
        self.model_manager = None
        self.setup_page()
        self.initialize_services()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="Fake News Detector",
            page_icon="🚨",
            layout="wide"
        )
        
        st.title("🚨 Fake News Detector")
        st.markdown("AI-powered system to detect and analyze potentially misleading news content")
    
    def initialize_services(self):
        """Initialize the fake news detection services"""
        if 'detector' not in st.session_state:
            # Initialize services
            text_processor = AdvancedTextProcessor()
            feature_extractors = [
                LinguisticFeatureExtractor(),
                SemanticFeatureExtractor()
            ]
            credibility_checker = SourceCredibilityChecker()
            fact_checker = APIFactChecker()
            
            # Initialize detector
            st.session_state.detector = FakeNewsDetector(
                text_processor=text_processor,
                feature_extractors=feature_extractors,
                credibility_checker=credibility_checker,
                fact_checker=fact_checker
            )
            
            # Add default classifiers
            from services import MLClassifier
            st.session_state.detector.add_classifier(MLClassifier("Random Forest", ModelType.RANDOM_FOREST))
            st.session_state.detector.add_classifier(MLClassifier("SVM", ModelType.SVM))
            st.session_state.detector.add_classifier(MLClassifier("Logistic Regression", ModelType.LOGISTIC_REGRESSION))
            
            # Initialize pipeline and model manager
            st.session_state.pipeline = DetectionPipeline(st.session_state.detector)
            st.session_state.model_manager = ModelManager(st.session_state.detector)
        
        self.detector = st.session_state.detector
        self.pipeline = st.session_state.pipeline
        self.model_manager = st.session_state.model_manager
    
    def render_sidebar(self):
        """Render the sidebar configuration"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            st.subheader("Detection Models")
            available_models = st.multiselect(
                "Select models to use:",
                ["Random Forest", "SVM", "Logistic Regression", "Naive Bayes"],
                default=["Random Forest", "SVM"]
            )
            
            # Confidence threshold
            st.subheader("Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1
            )
            
            # Update detector settings
            self.detector.min_confidence_threshold = confidence_threshold
            
            # Model management
            st.subheader("Model Management")
            if st.button("🔄 Refresh Models"):
                st.rerun()
            
            if st.button("💾 Save Models"):
                self.save_models()
            
            if st.button("📁 Load Models"):
                self.load_models()
    
    def render_main_content(self):
        """Render the main content area"""
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Single Article", "📊 Batch Analysis", "📈 Analytics", 
            "🤖 Model Training", "⚙️ Settings"
        ])
        
        with tab1:
            self.render_single_article_tab()
        
        with tab2:
            self.render_batch_analysis_tab()
        
        with tab3:
            self.render_analytics_tab()
        
        with tab4:
            self.render_model_training_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_single_article_tab(self):
        """Render single article analysis tab"""
        st.header("🔍 Single Article Analysis")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "URL Input", "File Upload"]
        )
        
        article = None
        
        if input_method == "Text Input":
            article = self.get_article_from_text()
        elif input_method == "URL Input":
            article = self.get_article_from_url()
        elif input_method == "File Upload":
            article = self.get_article_from_file()
        
        if article:
            # Check if models are trained
            trained_models = [c for c in self.detector.classifiers if c.is_trained]
            if not trained_models:
                st.warning("⚠️ No trained models available. Using basic heuristics for analysis.")
                st.info("💡 To improve accuracy, train models with your data in the 'Model Training' tab.")
            
            # Analyze article
            if st.button("🚨 Analyze Article", type="primary"):
                with st.spinner("Analyzing article..."):
                    result = self.pipeline.process_article(article)
                    self.display_detection_result(result)
    
    def get_article_from_text(self) -> Article:
        """Get article from text input"""
        title = st.text_input("Article Title:", placeholder="Enter the article title")
        content = st.text_area("Article Content:", placeholder="Enter the article content", height=200)
        source = st.text_input("Source:", placeholder="e.g., CNN, BBC, etc.")
        url = st.text_input("URL (optional):", placeholder="https://example.com/article")
        
        if title and content and source:
            return Article(
                title=title,
                content=content,
                source=source,
                url=url or "https://example.com",
                published_at=datetime.now()
            )
        return None
    
    def get_article_from_url(self) -> Article:
        """Get article from URL input"""
        url = st.text_input("Article URL:", placeholder="https://example.com/article")
        
        if url:
            if st.button("📥 Extract Article"):
                with st.spinner("Extracting article from URL..."):
                    # This would integrate with a web scraping service
                    st.info("URL extraction not implemented yet. Please use text input.")
        return None
    
    def get_article_from_file(self) -> Article:
        """Get article from file upload"""
        uploaded_file = st.file_uploader("Upload Article File:", type=['txt', 'csv'])
        
        if uploaded_file:
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
                return Article(
                    title="Uploaded Article",
                    content=content,
                    source="File Upload",
                    url="file://uploaded",
                    published_at=datetime.now()
                )
            elif uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
                st.info("CSV processing not implemented yet. Please use text input.")
        return None
    
    def display_detection_result(self, result):
        """Display detection result"""
        st.markdown("## 📊 Detection Results")
        
        # Main result
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🔴" if result.is_fake else "🟢"
            st.metric("Detection", f"{status_color} {'FAKE' if result.is_fake else 'REAL'}")
        
        with col2:
            confidence_color = "red" if result.confidence > 0.7 else "orange" if result.confidence > 0.5 else "green"
            st.metric("Confidence", f"{result.confidence:.2f}", delta=None)
        
        with col3:
            st.metric("Credibility", f"{result.credibility_score:.2f}")
        
        with col4:
            status_emoji = {"real": "🟢", "fake": "🔴", "uncertain": "🟡"}
            st.metric("Status", f"{status_emoji.get(result.detection_status.value, '❓')} {result.detection_status.value.upper()}")
        
        # Detailed analysis
        st.markdown("### 🔍 Detailed Analysis")
        
        # Reasoning
        st.markdown("**Reasoning:**")
        for reason in result.reasoning:
            st.write(f"• {reason}")
        
        # Model predictions
        if result.model_predictions:
            st.markdown("**Model Predictions:**")
            pred_df = pd.DataFrame([
                {
                    "Model": name,
                    "Prediction": "FAKE" if pred["is_fake"] else "REAL",
                    "Confidence": f"{pred['confidence']:.3f}"
                }
                for name, pred in result.model_predictions.items()
            ])
            st.dataframe(pred_df, use_container_width=True)
        
        # Features
        if result.features:
            st.markdown("**Key Features:**")
            feature_df = pd.DataFrame([
                {"Feature": name, "Value": f"{value:.3f}"}
                for name, value in sorted(result.features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            ])
            st.dataframe(feature_df, use_container_width=True)
    
    def render_batch_analysis_tab(self):
        """Render batch analysis tab"""
        st.header("📊 Batch Analysis")
        
        # File upload for batch processing
        uploaded_file = st.file_uploader(
            "Upload CSV file with articles:",
            type=['csv'],
            help="CSV should have columns: title, content, source, url"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            
            if st.button("🚨 Analyze All Articles", type="primary"):
                # Convert to articles
                articles = []
                for _, row in df.iterrows():
                    article = Article(
                        title=row.get('title', ''),
                        content=row.get('content', ''),
                        source=row.get('source', ''),
                        url=row.get('url', 'https://example.com'),
                        published_at=datetime.now()
                    )
                    articles.append(article)
                
                # Process articles
                with st.spinner(f"Analyzing {len(articles)} articles..."):
                    results = self.pipeline.process_batch(articles)
                
                # Display results
                self.display_batch_results(results)
    
    def display_batch_results(self, results):
        """Display batch analysis results"""
        st.markdown("## 📊 Batch Analysis Results")
        
        # Summary statistics
        stats = self.detector.get_detection_statistics(results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", stats.get('total_articles', 0))
        with col2:
            st.metric("Fake Articles", stats.get('fake_articles', 0))
        with col3:
            st.metric("Fake Percentage", f"{stats.get('fake_percentage', 0):.1f}%")
        with col4:
            st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.3f}")
        
        # Results table
        results_data = []
        for result in results:
            results_data.append({
                "Title": result.article.title[:50] + "..." if len(result.article.title) > 50 else result.article.title,
                "Source": result.article.source,
                "Detection": "🔴 FAKE" if result.is_fake else "🟢 REAL",
                "Confidence": f"{result.confidence:.3f}",
                "Credibility": f"{result.credibility_score:.3f}",
                "Status": result.detection_status.value.upper()
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            csv,
            file_name=f"fake_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.header("📈 Analytics Dashboard")
        
        if not self.pipeline.processing_history:
            st.info("No analysis data available. Please analyze some articles first.")
            return
        
        # Get statistics
        stats = self.pipeline.get_pipeline_statistics()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            status_data = stats.get('status_distribution', {})
            if status_data:
                fig = px.pie(
                    values=list(status_data.values()),
                    names=list(status_data.keys()),
                    title="Detection Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            confidences = [r.confidence for r in self.pipeline.processing_history]
            fig = px.histogram(
                x=confidences,
                nbins=20,
                title="Confidence Score Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if self.detector.classifiers:
            st.markdown("### 🎯 Feature Importance")
            for classifier in self.detector.get_active_classifiers():
                if hasattr(classifier, 'model') and hasattr(classifier.model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': classifier.feature_names,
                        'Importance': classifier.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f"Top Features - {classifier.name}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_model_training_tab(self):
        """Render model training tab"""
        st.header("🤖 Model Training")
        
        st.markdown("### 📊 Training Data")
        
        # Sample data upload
        training_file = st.file_uploader(
            "Upload training data (CSV):",
            type=['csv'],
            help="CSV should have columns: title, content, source, url, is_fake"
        )
        
        if training_file:
            df = pd.read_csv(training_file)
            st.dataframe(df.head())
            
            # Convert to training format
            training_data = []
            for _, row in df.iterrows():
                article = Article(
                    title=row.get('title', ''),
                    content=row.get('content', ''),
                    source=row.get('source', ''),
                    url=row.get('url', 'https://example.com'),
                    published_at=datetime.now()
                )
                is_fake = bool(row.get('is_fake', False))
                training_data.append((article, is_fake))
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models..."):
                    results = self.model_manager.train_all_models(training_data)
                
                # Display training results
                st.markdown("### 📊 Training Results")
                for model_name, result in results.items():
                    with st.expander(f"Model: {model_name}"):
                        if 'error' in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            st.metric("Training Accuracy", f"{result['training_accuracy']:.3f}")
                            if 'cross_validation' in result:
                                cv = result['cross_validation']
                                st.metric("CV Mean", f"{cv['cv_mean']:.3f}")
                                st.metric("CV Std", f"{cv['cv_std']:.3f}")
    
    def render_settings_tab(self):
        """Render settings tab"""
        st.header("⚙️ Settings")
        
        # Model configuration
        st.markdown("### 🤖 Model Configuration")
        
        # Add new models
        st.markdown("**Add New Models:**")
        model_type = st.selectbox("Model Type:", [mt.value for mt in ModelType])
        model_name = st.text_input("Model Name:", placeholder="my_model")
        
        if st.button("➕ Add Model"):
            if model_name:
                try:
                    mt = ModelType(model_type)
                    self.model_manager.add_model(mt, model_name)
                    st.success(f"Added {model_name} model")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding model: {e}")
        
        # Current models
        st.markdown("**Current Models:**")
        for classifier in self.detector.classifiers:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"• {classifier.name} ({classifier.model_type.value})")
            with col2:
                st.write(f"Trained: {'✅' if classifier.is_trained else '❌'}")
            with col3:
                if st.button(f"🗑️ Remove", key=f"remove_{classifier.name}"):
                    self.detector.remove_classifier(classifier.name)
                    st.rerun()
    
    def save_models(self):
        """Save models to disk"""
        try:
            self.detector.save_models("models")
            st.success("Models saved successfully!")
        except Exception as e:
            st.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load models from disk"""
        try:
            self.detector.load_models("models")
            st.success("Models loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def run(self):
        """Run the Streamlit application"""
        # Render sidebar
        self.render_sidebar()
        
        # Render main content
        self.render_main_content()


def main():
    """Main function to run the application"""
    app = FakeNewsApp()
    app.run()


if __name__ == "__main__":
    main()
