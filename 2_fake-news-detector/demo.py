#!/usr/bin/env python3
"""
Demo script for Fake News Detector using Object-Oriented Programming
"""

import os
from datetime import datetime
from dotenv import load_dotenv

from fake_news_detector import FakeNewsDetector, DetectionPipeline, ModelManager
from models import Article, ModelType, DetectionStatus, ConfidenceLevel
from services import (
    AdvancedTextProcessor, LinguisticFeatureExtractor, SemanticFeatureExtractor,
    MLClassifier, SourceCredibilityChecker, APIFactChecker
)


class FakeNewsDemo:
    """Demo class for the Fake News Detector"""
    
    def __init__(self):
        load_dotenv()
        self.detector = None
        self.pipeline = None
        self.model_manager = None
        self.initialize_services()
    
    def initialize_services(self):
        """Initialize the fake news detection services"""
        print("🔄 Initializing services...")
        
        # Initialize services
        text_processor = AdvancedTextProcessor()
        feature_extractors = [
            LinguisticFeatureExtractor(),
            SemanticFeatureExtractor()
        ]
        credibility_checker = SourceCredibilityChecker()
        fact_checker = APIFactChecker()
        
        # Initialize detector
        self.detector = FakeNewsDetector(
            text_processor=text_processor,
            feature_extractors=feature_extractors,
            credibility_checker=credibility_checker,
            fact_checker=fact_checker
        )
        
        # Initialize pipeline and model manager
        self.pipeline = DetectionPipeline(self.detector)
        self.model_manager = ModelManager(self.detector)
        
        print("✅ Services initialized successfully!")
    
    def create_sample_articles(self):
        """Create sample articles for testing"""
        sample_articles = [
            # Real news
            Article(
                title="Scientists Discover New Planet in Habitable Zone",
                content="A team of astronomers has discovered a new exoplanet located in the habitable zone of a nearby star. The planet, named Kepler-452b, is approximately 1.6 times the size of Earth and orbits a star similar to our Sun. The discovery was made using data from NASA's Kepler space telescope and has been confirmed by ground-based observations. This finding represents a significant step forward in the search for potentially habitable worlds beyond our solar system.",
                source="NASA",
                url="https://nasa.gov/news/kepler-452b-discovery",
                published_at=datetime.now()
            ),
            # Fake news
            Article(
                title="BREAKING: Aliens Contact Earth, Government Hiding Truth",
                content="URGENT: Multiple sources confirm that alien spacecraft have been detected near Earth's orbit. Government officials are reportedly covering up the truth from the public. A whistleblower from NASA claims that the agency has been in contact with extraterrestrial beings for over a decade. The aliens are said to be peaceful and want to share advanced technology with humanity. This information is being suppressed by the mainstream media and government agencies.",
                source="Conspiracy News Network",
                url="https://conspiracy-news.com/aliens-contact-earth",
                published_at=datetime.now()
            ),
            # Uncertain news
            Article(
                title="New Study Suggests Coffee May Have Health Benefits",
                content="A recent study published in a medical journal suggests that moderate coffee consumption may be associated with certain health benefits. The research, conducted over a period of five years, involved over 10,000 participants. However, the study authors caution that more research is needed to confirm these findings. The results are preliminary and should not be considered definitive medical advice.",
                source="Health News Daily",
                url="https://healthnews.com/coffee-benefits-study",
                published_at=datetime.now()
            )
        ]
        
        return sample_articles
    
    def create_training_data(self):
        """Create sample training data"""
        training_articles = [
            # Real news examples
            (Article(
                title="Economic Growth Exceeds Expectations",
                content="The latest economic data shows that GDP growth has exceeded expectations for the third consecutive quarter. The strong performance is attributed to increased consumer spending and business investment.",
                source="Financial Times",
                url="https://ft.com/economic-growth",
                published_at=datetime.now()
            ), False),
            
            (Article(
                title="Climate Change Summit Reaches Agreement",
                content="World leaders have reached a historic agreement on climate change at the international summit. The agreement includes commitments to reduce carbon emissions and invest in renewable energy.",
                source="BBC News",
                url="https://bbc.com/climate-summit",
                published_at=datetime.now()
            ), False),
            
            # Fake news examples
            (Article(
                title="Vaccines Cause Autism - New Study Proves It",
                content="A groundbreaking new study has definitively proven that vaccines cause autism in children. The research, conducted by independent scientists, shows a direct correlation between vaccination and autism spectrum disorders.",
                source="Natural Health News",
                url="https://naturalhealth.com/vaccines-autism",
                published_at=datetime.now()
            ), True),
            
            (Article(
                title="Moon Landing Was Faked - NASA Admits It",
                content="NASA has finally admitted that the 1969 moon landing was staged in a Hollywood studio. A former NASA employee has come forward with evidence that the entire Apollo program was a hoax designed to win the space race.",
                source="Truth Seekers",
                url="https://truthseekers.com/moon-landing-hoax",
                published_at=datetime.now()
            ), True)
        ]
        
        return training_articles
    
    def test_text_processing(self):
        """Test text processing capabilities"""
        print("\n🧪 Testing text processing...")
        
        sample_text = "This is a SAMPLE text with HTML <b>tags</b> and URLs https://example.com for testing!"
        
        try:
            # Test preprocessing
            processed = self.detector.text_processor.preprocess(sample_text)
            print(f"✅ Preprocessing: {processed}")
            
            # Test tokenization
            tokens = self.detector.text_processor.tokenize(processed)
            print(f"✅ Tokenization: {len(tokens)} tokens")
            
            # Test normalization
            normalized = self.detector.text_processor.normalize(processed)
            print(f"✅ Normalization: {normalized}")
            
        except Exception as e:
            print(f"❌ Text processing test failed: {e}")
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        print("\n🔍 Testing feature extraction...")
        
        sample_text = "This is a sample article about technology and innovation in the modern world."
        
        try:
            # Test linguistic features
            linguistic_extractor = LinguisticFeatureExtractor()
            linguistic_features = linguistic_extractor.extract_features(sample_text)
            print(f"✅ Linguistic features: {len(linguistic_features)} features")
            print(f"   Key features: {list(linguistic_features.keys())[:5]}")
            
            # Test semantic features
            semantic_extractor = SemanticFeatureExtractor()
            semantic_features = semantic_extractor.extract_features(sample_text)
            print(f"✅ Semantic features: {len(semantic_features)} features")
            print(f"   Key features: {list(semantic_features.keys())[:5]}")
            
        except Exception as e:
            print(f"❌ Feature extraction test failed: {e}")
    
    def test_model_training(self):
        """Test model training"""
        print("\n🤖 Testing model training...")
        
        # Create training data
        training_data = self.create_training_data()
        
        # Add classifiers
        self.model_manager.add_model(ModelType.RANDOM_FOREST, "Random Forest")
        self.model_manager.add_model(ModelType.SVM, "SVM")
        self.model_manager.add_model(ModelType.LOGISTIC_REGRESSION, "Logistic Regression")
        
        try:
            # Train models
            results = self.model_manager.train_all_models(training_data)
            
            print("✅ Model training results:")
            for model_name, result in results.items():
                if 'error' in result:
                    print(f"   ❌ {model_name}: {result['error']}")
                else:
                    print(f"   ✅ {model_name}: Accuracy = {result['training_accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ Model training test failed: {e}")
    
    def test_detection(self):
        """Test fake news detection"""
        print("\n🚨 Testing fake news detection...")
        
        # Create sample articles
        articles = self.create_sample_articles()
        
        try:
            # Process each article
            for i, article in enumerate(articles, 1):
                print(f"\n📰 Article {i}: {article.title}")
                print(f"   Source: {article.source}")
                
                # Detect fake news
                result = self.pipeline.process_article(article)
                
                # Display results
                status_emoji = "🔴" if result.is_fake else "🟢"
                print(f"   Detection: {status_emoji} {'FAKE' if result.is_fake else 'REAL'}")
                print(f"   Confidence: {result.confidence:.3f}")
                print(f"   Credibility: {result.credibility_score:.3f}")
                print(f"   Status: {result.detection_status.value.upper()}")
                
                # Show reasoning
                if result.reasoning:
                    print(f"   Reasoning: {'; '.join(result.reasoning[:2])}")
            
        except Exception as e:
            print(f"❌ Detection test failed: {e}")
    
    def test_batch_processing(self):
        """Test batch processing"""
        print("\n📊 Testing batch processing...")
        
        # Create sample articles
        articles = self.create_sample_articles()
        
        try:
            # Process batch
            results = self.pipeline.process_batch(articles)
            
            # Get statistics
            stats = self.detector.get_detection_statistics(results)
            
            print("✅ Batch processing results:")
            print(f"   Total articles: {stats.get('total_articles', 0)}")
            print(f"   Fake articles: {stats.get('fake_articles', 0)}")
            print(f"   Fake percentage: {stats.get('fake_percentage', 0):.1f}%")
            print(f"   Average confidence: {stats.get('average_confidence', 0):.3f}")
            
        except Exception as e:
            print(f"❌ Batch processing test failed: {e}")
    
    def test_model_comparison(self):
        """Test model comparison"""
        print("\n⚖️ Testing model comparison...")
        
        # Create test data
        test_data = self.create_training_data()
        
        try:
            # Compare models
            comparison_results = self.model_manager.compare_models(test_data)
            
            print("✅ Model comparison results:")
            for model_name, result in comparison_results.items():
                if 'error' in result:
                    print(f"   ❌ {model_name}: {result['error']}")
                else:
                    print(f"   ✅ {model_name}: F1 = {result['f1_score']:.3f}, Accuracy = {result['accuracy']:.3f}")
            
            # Get best model
            best_model = self.model_manager.get_best_model(test_data)
            if best_model:
                print(f"   🏆 Best model: {best_model.name}")
            
        except Exception as e:
            print(f"❌ Model comparison test failed: {e}")
    
    def run_full_demo(self):
        """Run the complete demo"""
        print("🚨 Fake News Detector - OOP Demo")
        print("=" * 50)
        
        # Test individual components
        self.test_text_processing()
        self.test_feature_extraction()
        self.test_model_training()
        self.test_detection()
        self.test_batch_processing()
        self.test_model_comparison()
        
        print("\n🎉 Demo completed successfully!")
        print("\n📝 Next steps:")
        print("1. Start the web interface: streamlit run app.py")
        print("2. Train models with your own data")
        print("3. Analyze real news articles")
        print("4. Explore the analytics dashboard")
    
    def run_quick_test(self):
        """Run a quick test without full training"""
        print("🧪 Quick Test")
        print("=" * 30)
        
        try:
            # Test basic functionality
            from models import Article, DetectionStatus
            from services import AdvancedTextProcessor, LinguisticFeatureExtractor
            
            # Test article creation
            article = Article(
                title="Test Article",
                content="This is a test article for fake news detection.",
                source="Test Source",
                url="https://test.com",
                published_at=datetime.now()
            )
            print("✅ Article model works")
            
            # Test text processor
            processor = AdvancedTextProcessor()
            processed = processor.preprocess("Test text with HTML <b>tags</b>")
            print("✅ Text processor works")
            
            # Test feature extractor
            extractor = LinguisticFeatureExtractor()
            features = extractor.extract_features("Test text for feature extraction")
            print("✅ Feature extractor works")
            
            print("\n🎉 All components working correctly!")
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")


def main():
    """Main function"""
    demo = FakeNewsDemo()
    
    # Check if we have training data
    if os.path.exists("training_data.csv"):
        print("🚀 Running full demo with training data...")
        demo.run_full_demo()
    else:
        print("🧪 Running quick test without training data...")
        demo.run_quick_test()
        
        print("\n💡 To run the full demo:")
        print("1. Create a training_data.csv file with columns: title, content, source, url, is_fake")
        print("2. Run: python demo.py")


if __name__ == "__main__":
    main()
