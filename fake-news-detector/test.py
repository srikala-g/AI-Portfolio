#!/usr/bin/env python3
"""
Test suite for Fake News Detector using Object-Oriented Programming
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FakeNewsTestSuite:
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
            import numpy as np
            print("✅ numpy imported successfully")
            
            import pandas as pd
            print("✅ pandas imported successfully")
            
            import sklearn
            print("✅ scikit-learn imported successfully")
            
            import nltk
            print("✅ nltk imported successfully")
            
            import textstat
            print("✅ textstat imported successfully")
            
            from textblob import TextBlob
            print("✅ textblob imported successfully")
            
            import streamlit as st
            print("✅ streamlit imported successfully")
            
            import plotly.express as px
            print("✅ plotly imported successfully")
            
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
                Article, DetectionResult, DetectionStatus, ConfidenceLevel,
                ModelType, TextProcessor, FeatureExtractor, Classifier
            )
            
            # Test Article creation
            article = Article(
                title="Test Article",
                content="This is a test article for fake news detection.",
                source="Test Source",
                url="https://test.com",
                published_at=datetime.now()
            )
            print("✅ Article model works")
            
            # Test DetectionResult
            result = DetectionResult(
                article=article,
                is_fake=False,
                confidence=0.8,
                credibility_score=0.9,
                reasoning=["Test reasoning"],
                features={"test": 0.5},
                model_predictions={"test_model": {"is_fake": False, "confidence": 0.8}},
                detection_status=DetectionStatus.REAL,
                confidence_level=ConfidenceLevel.HIGH
            )
            print("✅ DetectionResult model works")
            
            # Test enums
            assert DetectionStatus.REAL.value == "real"
            assert ConfidenceLevel.HIGH.value == "high"
            assert ModelType.RANDOM_FOREST.value == "random_forest"
            print("✅ Enums work correctly")
            
            return True
            
        except Exception as e:
            print(f"❌ Models test failed: {e}")
            return False
    
    def test_services(self):
        """Test service classes"""
        try:
            from services import (
                AdvancedTextProcessor, LinguisticFeatureExtractor, 
                SemanticFeatureExtractor, MLClassifier, SourceCredibilityChecker
            )
            from models import ModelType, Article
            
            # Test text processor
            processor = AdvancedTextProcessor()
            processed = processor.preprocess("Test text with HTML <b>tags</b>")
            print("✅ AdvancedTextProcessor works")
            
            # Test feature extractors
            linguistic_extractor = LinguisticFeatureExtractor()
            features = linguistic_extractor.extract_features("Test text for feature extraction")
            print("✅ LinguisticFeatureExtractor works")
            
            semantic_extractor = SemanticFeatureExtractor()
            semantic_features = semantic_extractor.extract_features("Test text for semantic analysis")
            print("✅ SemanticFeatureExtractor works")
            
            # Test classifier
            classifier = MLClassifier("test_classifier", ModelType.RANDOM_FOREST)
            print("✅ MLClassifier works")
            
            # Test credibility checker
            credibility_checker = SourceCredibilityChecker()
            article = Article(
                title="Test",
                content="Test content",
                source="BBC",
                url="https://bbc.com/test",
                published_at=datetime.now()
            )
            credibility = credibility_checker.check_credibility(article)
            print("✅ SourceCredibilityChecker works")
            
            return True
            
        except Exception as e:
            print(f"❌ Services test failed: {e}")
            return False
    
    def test_fake_news_detector(self):
        """Test fake news detector"""
        try:
            from fake_news_detector import FakeNewsDetector, DetectionPipeline, ModelManager
            from services import (
                AdvancedTextProcessor, LinguisticFeatureExtractor, 
                SemanticFeatureExtractor, SourceCredibilityChecker
            )
            from models import Article, ModelType
            
            # Test detector creation
            detector = FakeNewsDetector()
            print("✅ FakeNewsDetector works")
            
            # Test pipeline
            pipeline = DetectionPipeline(detector)
            print("✅ DetectionPipeline works")
            
            # Test model manager
            model_manager = ModelManager(detector)
            print("✅ ModelManager works")
            
            # Test article processing
            article = Article(
                title="Test Article",
                content="This is a test article for fake news detection.",
                source="Test Source",
                url="https://test.com",
                published_at=datetime.now()
            )
            
            # This will fail without trained models, but we can test the structure
            try:
                result = detector.detect_fake_news(article)
                print("✅ Detection works")
            except ValueError as e:
                if "No classifiers available" in str(e):
                    print("✅ Detection structure works (no trained models)")
                else:
                    raise e
            
            return True
            
        except Exception as e:
            print(f"❌ FakeNewsDetector test failed: {e}")
            return False
    
    def test_text_processing(self):
        """Test text processing functionality"""
        try:
            from services import AdvancedTextProcessor
            
            processor = AdvancedTextProcessor()
            
            # Test preprocessing
            text = "This is a SAMPLE text with HTML <b>tags</b> and URLs https://example.com!"
            processed = processor.preprocess(text)
            assert "<b>" not in processed
            assert "https://example.com" not in processed
            print("✅ Text preprocessing works")
            
            # Test tokenization
            tokens = processor.tokenize(processed)
            assert isinstance(tokens, list)
            print("✅ Text tokenization works")
            
            # Test normalization
            normalized = processor.normalize(processed)
            assert isinstance(normalized, str)
            print("✅ Text normalization works")
            
            return True
            
        except Exception as e:
            print(f"❌ Text processing test failed: {e}")
            return False
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        try:
            from services import LinguisticFeatureExtractor, SemanticFeatureExtractor
            
            # Test linguistic features
            linguistic_extractor = LinguisticFeatureExtractor()
            features = linguistic_extractor.extract_features("This is a test article about technology and innovation.")
            
            assert isinstance(features, dict)
            assert 'word_count' in features
            assert 'char_count' in features
            assert 'polarity' in features
            print("✅ Linguistic feature extraction works")
            
            # Test semantic features
            semantic_extractor = SemanticFeatureExtractor()
            semantic_features = semantic_extractor.extract_features("This is a test article about technology and innovation.")
            
            assert isinstance(semantic_features, dict)
            print("✅ Semantic feature extraction works")
            
            return True
            
        except Exception as e:
            print(f"❌ Feature extraction test failed: {e}")
            return False
    
    def test_classifier(self):
        """Test classifier functionality"""
        try:
            from services import MLClassifier
            from models import ModelType
            
            # Test classifier creation
            classifier = MLClassifier("test_classifier", ModelType.RANDOM_FOREST)
            assert classifier.name == "test_classifier"
            assert classifier.model_type == ModelType.RANDOM_FOREST
            print("✅ Classifier creation works")
            
            # Test model initialization
            assert classifier.model is not None
            print("✅ Model initialization works")
            
            return True
            
        except Exception as e:
            print(f"❌ Classifier test failed: {e}")
            return False
    
    def test_credibility_checking(self):
        """Test credibility checking"""
        try:
            from services import SourceCredibilityChecker
            from models import Article
            
            checker = SourceCredibilityChecker()
            
            # Test credible source
            credible_article = Article(
                title="Test",
                content="Test content",
                source="BBC",
                url="https://bbc.com/test",
                published_at=datetime.now()
            )
            credibility = checker.check_credibility(credible_article)
            assert 0.0 <= credibility <= 1.0
            print("✅ Credibility checking works")
            
            # Test credibility factors
            factors = checker.get_credibility_factors(credible_article)
            assert isinstance(factors, dict)
            print("✅ Credibility factors work")
            
            return True
            
        except Exception as e:
            print(f"❌ Credibility checking test failed: {e}")
            return False
    
    def test_app_imports(self):
        """Test app imports"""
        try:
            from app import FakeNewsApp
            print("✅ Streamlit app imports work")
            
            return True
            
        except Exception as e:
            print(f"❌ App imports test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚨 Fake News Detector - OOP Test Suite")
        print("=" * 50)
        
        tests = [
            ("Import Test", self.test_imports),
            ("Models Test", self.test_models),
            ("Services Test", self.test_services),
            ("FakeNewsDetector Test", self.test_fake_news_detector),
            ("Text Processing Test", self.test_text_processing),
            ("Feature Extraction Test", self.test_feature_extraction),
            ("Classifier Test", self.test_classifier),
            ("Credibility Checking Test", self.test_credibility_checking),
            ("App Imports Test", self.test_app_imports)
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print results
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {self.passed_tests}/{self.total_tests} tests passed")
        
        if self.passed_tests == self.total_tests:
            print("🎉 All tests passed! The fake news detector is ready to use.")
            print("\n📝 Next steps:")
            print("1. Install missing dependencies: pip install -r requirements.txt")
            print("2. Run the demo: python demo.py")
            print("3. Start the web interface: streamlit run app.py")
        else:
            print("❌ Some tests failed. Please check the output above for details.")
            print("\n🔧 To fix issues:")
            print("1. Install missing packages: pip install -r requirements.txt")
            print("2. Check your Python environment")
            print("3. Ensure all dependencies are installed")
        
        return self.passed_tests == self.total_tests


def main():
    """Main function"""
    test_suite = FakeNewsTestSuite()
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
