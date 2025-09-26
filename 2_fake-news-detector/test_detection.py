#!/usr/bin/env python3
"""
Test script to verify fake news detection works
"""

from datetime import datetime
from models import Article
from fake_news_detector import FakeNewsDetector, DetectionPipeline
from services import (
    AdvancedTextProcessor, LinguisticFeatureExtractor, SemanticFeatureExtractor,
    SourceCredibilityChecker, APIFactChecker
)

def test_detection():
    """Test fake news detection"""
    print("🧪 Testing Fake News Detection")
    print("=" * 40)
    
    # Initialize detector
    text_processor = AdvancedTextProcessor()
    feature_extractors = [
        LinguisticFeatureExtractor(),
        SemanticFeatureExtractor()
    ]
    credibility_checker = SourceCredibilityChecker()
    fact_checker = APIFactChecker()
    
    detector = FakeNewsDetector(
        text_processor=text_processor,
        feature_extractors=feature_extractors,
        credibility_checker=credibility_checker,
        fact_checker=fact_checker
    )
    
    # Create test articles
    real_article = Article(
        title="Scientists Discover New Planet in Habitable Zone",
        content="A team of astronomers has discovered a new exoplanet located in the habitable zone of a nearby star. The planet, named Kepler-452b, is approximately 1.6 times the size of Earth and orbits a star similar to our Sun.",
        source="NASA",
        url="https://nasa.gov/news/kepler-452b-discovery",
        published_at=datetime.now()
    )
    
    fake_article = Article(
        title="BREAKING: Aliens Contact Earth, Government Hiding Truth!!!",
        content="URGENT: Multiple sources confirm that alien spacecraft have been detected near Earth's orbit. Government officials are reportedly covering up the truth from the public.",
        source="Conspiracy News Network",
        url="https://conspiracy-news.com/aliens-contact-earth",
        published_at=datetime.now()
    )
    
    # Test detection
    print("📰 Testing Real Article:")
    print(f"   Title: {real_article.title}")
    print(f"   Source: {real_article.source}")
    
    try:
        result = detector.detect_fake_news(real_article)
        print(f"   Detection: {'🔴 FAKE' if result.is_fake else '🟢 REAL'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Credibility: {result.credibility_score:.3f}")
        print(f"   Status: {result.detection_status.value.upper()}")
        if result.reasoning:
            print(f"   Reasoning: {'; '.join(result.reasoning[:2])}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n📰 Testing Fake Article:")
    print(f"   Title: {fake_article.title}")
    print(f"   Source: {fake_article.source}")
    
    try:
        result = detector.detect_fake_news(fake_article)
        print(f"   Detection: {'🔴 FAKE' if result.is_fake else '🟢 REAL'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Credibility: {result.credibility_score:.3f}")
        print(f"   Status: {result.detection_status.value.upper()}")
        if result.reasoning:
            print(f"   Reasoning: {'; '.join(result.reasoning[:2])}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Detection test completed!")

if __name__ == "__main__":
    test_detection()
