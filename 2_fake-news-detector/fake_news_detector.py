"""
Main Fake News Detector class using Object-Oriented Programming
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from models import (
    Article, DetectionResult, DetectionStatus, ConfidenceLevel,
    TextProcessor, FeatureExtractor, Classifier, CredibilityChecker, 
    FactChecker, ModelEvaluator, ModelType
)
from services import (
    AdvancedTextProcessor, LinguisticFeatureExtractor, SemanticFeatureExtractor,
    MLClassifier, SourceCredibilityChecker, APIFactChecker, ModelEvaluatorImpl
)


class FakeNewsDetector:
    """
    Main fake news detector class that orchestrates all components
    using proper Object-Oriented Programming principles
    """
    
    def __init__(self, 
                 text_processor: Optional[TextProcessor] = None,
                 feature_extractors: Optional[List[FeatureExtractor]] = None,
                 classifiers: Optional[List[Classifier]] = None,
                 credibility_checker: Optional[CredibilityChecker] = None,
                 fact_checker: Optional[FactChecker] = None,
                 model_evaluator: Optional[ModelEvaluator] = None):
        """
        Initialize the fake news detector with all required services
        
        Args:
            text_processor: Text processing service
            feature_extractors: List of feature extraction services
            classifiers: List of classification models
            credibility_checker: Credibility checking service
            fact_checker: Fact checking service
            model_evaluator: Model evaluation service
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize services with defaults if not provided
        self.text_processor = text_processor or AdvancedTextProcessor()
        self.feature_extractors = feature_extractors or [
            LinguisticFeatureExtractor(),
            SemanticFeatureExtractor()
        ]
        self.classifiers = classifiers or []
        self.credibility_checker = credibility_checker or SourceCredibilityChecker()
        self.fact_checker = fact_checker or APIFactChecker()
        self.model_evaluator = model_evaluator or ModelEvaluatorImpl()
        
        # Detection settings
        self.min_confidence_threshold = 0.6
        self.ensemble_weights = {}  # Will be set based on model performance
    
    def add_classifier(self, classifier: Classifier) -> None:
        """Add a new classifier to the detector"""
        self.classifiers.append(classifier)
    
    def remove_classifier(self, classifier_name: str) -> bool:
        """Remove a classifier by name"""
        for i, classifier in enumerate(self.classifiers):
            if classifier.name == classifier_name:
                self.classifiers.pop(i)
                return True
        return False
    
    def get_active_classifiers(self) -> List[Classifier]:
        """Get all active (trained) classifiers"""
        return [classifier for classifier in self.classifiers if classifier.is_trained]
    
    def train_models(self, training_data: List[Tuple[Article, bool]]) -> Dict[str, Dict[str, float]]:
        """
        Train all classifiers with provided data
        
        Args:
            training_data: List of (Article, is_fake) tuples
            
        Returns:
            Dictionary of training results for each classifier
        """
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        # Prepare training data
        X, y = self._prepare_training_data(training_data)
        
        results = {}
        
        for classifier in self.classifiers:
            try:
                print(f"Training {classifier.name}...")
                classifier.train(X, y)
                
                # Evaluate the trained model
                evaluation = classifier.evaluate(X, y)
                results[classifier.name] = evaluation
                
                print(f"✅ {classifier.name} trained successfully")
                print(f"   Accuracy: {evaluation['accuracy']:.3f}")
                print(f"   F1-Score: {evaluation['f1_score']:.3f}")
                
            except Exception as e:
                print(f"❌ Error training {classifier.name}: {e}")
                results[classifier.name] = {"error": str(e)}
        
        # Set ensemble weights based on performance
        self._update_ensemble_weights(results)
        
        return results
    
    def _prepare_training_data(self, training_data: List[Tuple[Article, bool]]) -> Tuple[List[Dict[str, float]], List[bool]]:
        """Prepare training data by extracting features"""
        X = []
        y = []
        
        for article, is_fake in training_data:
            # Preprocess text
            processed_text = self.text_processor.preprocess(article.content)
            
            # Extract features
            features = self._extract_all_features(processed_text)
            X.append(features)
            y.append(is_fake)
        
        return X, y
    
    def _extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract features using all feature extractors"""
        all_features = {}
        
        for extractor in self.feature_extractors:
            try:
                features = extractor.extract_features(text)
                all_features.update(features)
            except Exception as e:
                print(f"Error extracting features with {extractor.name}: {e}")
        
        return all_features
    
    def _update_ensemble_weights(self, training_results: Dict[str, Dict[str, float]]) -> None:
        """Update ensemble weights based on model performance"""
        total_f1 = 0
        valid_models = 0
        
        for classifier_name, results in training_results.items():
            if 'f1_score' in results and not isinstance(results['f1_score'], str):
                total_f1 += results['f1_score']
                valid_models += 1
        
        if valid_models > 0:
            for classifier_name, results in training_results.items():
                if 'f1_score' in results and not isinstance(results['f1_score'], str):
                    self.ensemble_weights[classifier_name] = results['f1_score'] / total_f1
                else:
                    self.ensemble_weights[classifier_name] = 0.0
        else:
            # Equal weights if no valid models
            for classifier_name in training_results.keys():
                self.ensemble_weights[classifier_name] = 1.0 / len(training_results)
    
    def detect_fake_news(self, article: Article) -> DetectionResult:
        """
        Detect if an article contains fake news
        
        Args:
            article: Article to analyze
            
        Returns:
            DetectionResult with analysis results
        """
        if not self.classifiers:
            # Return a basic analysis without ML models
            return self._basic_analysis(article)
        
        # Preprocess text
        processed_text = self.text_processor.preprocess(article.content)
        
        # Extract features
        features = self._extract_all_features(processed_text)
        
        # Get predictions from all trained classifiers
        predictions = {}
        for classifier in self.get_active_classifiers():
            try:
                is_fake, confidence = classifier.predict(features)
                predictions[classifier.name] = {
                    'is_fake': is_fake,
                    'confidence': confidence
                }
            except Exception as e:
                print(f"Error with classifier {classifier.name}: {e}")
                predictions[classifier.name] = {
                    'is_fake': False,
                    'confidence': 0.5
                }
        
        # Ensemble prediction
        ensemble_result = self._ensemble_predictions(predictions)
        
        # Check credibility
        credibility_score = self.credibility_checker.check_credibility(article)
        
        # Fact check
        fact_check_result = self.fact_checker.check_facts(article)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            ensemble_result, credibility_score, fact_check_result, predictions
        )
        
        # Determine detection status
        detection_status = self._determine_detection_status(ensemble_result['confidence'])
        
        return DetectionResult(
            article=article,
            is_fake=ensemble_result['is_fake'],
            confidence=ensemble_result['confidence'],
            credibility_score=credibility_score,
            reasoning=reasoning,
            features=features,
            model_predictions=predictions,
            detection_status=detection_status,
            confidence_level=ConfidenceLevel.LOW  # Will be set in __post_init__
        )
    
    def _ensemble_predictions(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Combine predictions from multiple classifiers using weighted voting"""
        if not predictions:
            return {'is_fake': False, 'confidence': 0.5}
        
        weighted_fake_score = 0.0
        total_weight = 0.0
        
        for classifier_name, pred in predictions.items():
            weight = self.ensemble_weights.get(classifier_name, 1.0 / len(predictions))
            
            if pred['is_fake']:
                weighted_fake_score += weight * pred['confidence']
            else:
                weighted_fake_score += weight * (1 - pred['confidence'])
            
            total_weight += weight
        
        if total_weight > 0:
            final_confidence = weighted_fake_score / total_weight
        else:
            final_confidence = 0.5
        
        return {
            'is_fake': final_confidence > 0.5,
            'confidence': final_confidence
        }
    
    def _basic_analysis(self, article: Article) -> DetectionResult:
        """Perform basic analysis without trained models"""
        # Preprocess text
        processed_text = self.text_processor.preprocess(article.content)
        
        # Extract features
        features = self._extract_all_features(processed_text)
        
        # Basic heuristics
        is_fake = False
        confidence = 0.5
        reasoning = []
        
        # Check for common fake news indicators
        title_lower = article.title.lower()
        content_lower = article.content.lower()
        
        # Sensational language
        sensational_words = ['breaking', 'shocking', 'urgent', 'exclusive', 'revealed', 'exposed']
        if any(word in title_lower for word in sensational_words):
            is_fake = True
            confidence += 0.2
            reasoning.append("Sensational language detected in title")
        
        # Excessive punctuation
        if article.title.count('!') > 2 or article.title.count('?') > 2:
            is_fake = True
            confidence += 0.1
            reasoning.append("Excessive punctuation in title")
        
        # Check credibility
        credibility_score = self.credibility_checker.check_credibility(article)
        
        # Adjust confidence based on credibility
        if credibility_score < 0.3:
            is_fake = True
            confidence += 0.3
            reasoning.append("Low source credibility")
        elif credibility_score > 0.7:
            is_fake = False
            confidence += 0.2
            reasoning.append("High source credibility")
        
        # Check content length
        if len(article.content) < 100:
            is_fake = True
            confidence += 0.1
            reasoning.append("Very short content")
        
        # Normalize confidence
        confidence = min(confidence, 1.0)
        
        # Determine status
        if confidence >= 0.8:
            detection_status = DetectionStatus.FAKE if is_fake else DetectionStatus.REAL
        else:
            detection_status = DetectionStatus.UNCERTAIN
        
        return DetectionResult(
            article=article,
            is_fake=is_fake,
            confidence=confidence,
            credibility_score=credibility_score,
            reasoning=reasoning,
            features=features,
            model_predictions={},
            detection_status=detection_status,
            confidence_level=ConfidenceLevel.LOW
        )
    
    def _generate_reasoning(self, ensemble_result: Dict[str, float], 
                          credibility_score: float, fact_check_result: Dict[str, Any],
                          predictions: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate human-readable reasoning for the detection result"""
        reasoning = []
        
        # Model predictions reasoning
        fake_predictions = sum(1 for pred in predictions.values() if pred['is_fake'])
        total_predictions = len(predictions)
        
        if fake_predictions > total_predictions / 2:
            reasoning.append(f"{fake_predictions}/{total_predictions} models predicted fake news")
        else:
            reasoning.append(f"{total_predictions - fake_predictions}/{total_predictions} models predicted real news")
        
        # Confidence reasoning
        if ensemble_result['confidence'] > 0.8:
            reasoning.append("High confidence in prediction")
        elif ensemble_result['confidence'] > 0.6:
            reasoning.append("Moderate confidence in prediction")
        else:
            reasoning.append("Low confidence in prediction")
        
        # Credibility reasoning
        if credibility_score > 0.7:
            reasoning.append("Source appears credible")
        elif credibility_score < 0.3:
            reasoning.append("Source credibility is questionable")
        else:
            reasoning.append("Source credibility is uncertain")
        
        # Fact check reasoning
        if fact_check_result.get('fact_check_available'):
            reasoning.append("Fact-checking information available")
        else:
            reasoning.append("No fact-checking information available")
        
        return reasoning
    
    def _determine_detection_status(self, confidence: float) -> DetectionStatus:
        """Determine detection status based on confidence"""
        if confidence >= 0.8:
            return DetectionStatus.FAKE if confidence > 0.5 else DetectionStatus.REAL
        else:
            return DetectionStatus.UNCERTAIN
    
    def batch_detect(self, articles: List[Article]) -> List[DetectionResult]:
        """Detect fake news for multiple articles"""
        results = []
        
        for article in articles:
            try:
                result = self.detect_fake_news(article)
                results.append(result)
            except Exception as e:
                print(f"Error processing article '{article.title}': {e}")
                # Create a default result for failed articles
                results.append(DetectionResult(
                    article=article,
                    is_fake=False,
                    confidence=0.0,
                    credibility_score=0.5,
                    reasoning=[f"Error: {str(e)}"],
                    features={},
                    model_predictions={},
                    detection_status=DetectionStatus.UNCERTAIN,
                    confidence_level=ConfidenceLevel.LOW
                ))
        
        return results
    
    def get_detection_statistics(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """Get statistics from detection results"""
        if not results:
            return {}
        
        total_articles = len(results)
        fake_articles = sum(1 for r in results if r.is_fake)
        real_articles = total_articles - fake_articles
        
        avg_confidence = np.mean([r.confidence for r in results])
        avg_credibility = np.mean([r.credibility_score for r in results])
        
        status_counts = {}
        for status in DetectionStatus:
            status_counts[status.value] = sum(1 for r in results if r.detection_status == status)
        
        return {
            'total_articles': total_articles,
            'fake_articles': fake_articles,
            'real_articles': real_articles,
            'fake_percentage': (fake_articles / total_articles) * 100,
            'average_confidence': avg_confidence,
            'average_credibility': avg_credibility,
            'status_distribution': status_counts
        }
    
    def save_models(self, directory: str) -> None:
        """Save all trained models to directory"""
        os.makedirs(directory, exist_ok=True)
        
        for classifier in self.get_active_classifiers():
            filepath = os.path.join(directory, f"{classifier.name}.joblib")
            classifier.save_model(filepath)
            print(f"Saved {classifier.name} to {filepath}")
    
    def load_models(self, directory: str) -> None:
        """Load models from directory"""
        for classifier in self.classifiers:
            filepath = os.path.join(directory, f"{classifier.name}.joblib")
            if os.path.exists(filepath):
                classifier.load_model(filepath)
                print(f"Loaded {classifier.name} from {filepath}")


class DetectionPipeline:
    """
    Pipeline for processing articles through the detection system
    """
    
    def __init__(self, detector: FakeNewsDetector):
        self.detector = detector
        self.processing_history = []
    
    def process_article(self, article: Article) -> DetectionResult:
        """Process a single article through the pipeline"""
        result = self.detector.detect_fake_news(article)
        self.processing_history.append(result)
        return result
    
    def process_batch(self, articles: List[Article]) -> List[DetectionResult]:
        """Process multiple articles through the pipeline"""
        results = self.detector.batch_detect(articles)
        self.processing_history.extend(results)
        return results
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics from the processing pipeline"""
        if not self.processing_history:
            return {}
        
        return self.detector.get_detection_statistics(self.processing_history)
    
    def clear_history(self) -> None:
        """Clear processing history"""
        self.processing_history = []


class ModelManager:
    """
    Manager for handling multiple models and their lifecycle
    """
    
    def __init__(self, detector: FakeNewsDetector):
        self.detector = detector
        self.model_evaluator = ModelEvaluatorImpl()
    
    def add_model(self, model_type: ModelType, name: str) -> Classifier:
        """Add a new model to the detector"""
        classifier = MLClassifier(name, model_type)
        self.detector.add_classifier(classifier)
        return classifier
    
    def train_all_models(self, training_data: List[Tuple[Article, bool]]) -> Dict[str, Dict[str, float]]:
        """Train all models with cross-validation"""
        results = {}
        
        for classifier in self.detector.classifiers:
            try:
                # Prepare data
                X, y = self.detector._prepare_training_data(training_data)
                
                # Train model
                classifier.train(X, y)
                
                # Cross-validation
                cv_results = self.model_evaluator.cross_validate(classifier, X, y)
                
                # Feature importance
                feature_importance = self.model_evaluator.get_feature_importance(classifier)
                
                results[classifier.name] = {
                    'training_accuracy': classifier.accuracy,
                    'cross_validation': cv_results,
                    'feature_importance': feature_importance
                }
                
            except Exception as e:
                results[classifier.name] = {'error': str(e)}
        
        return results
    
    def compare_models(self, test_data: List[Tuple[Article, bool]]) -> Dict[str, Dict[str, float]]:
        """Compare performance of all models"""
        results = {}
        X, y = self.detector._prepare_training_data(test_data)
        
        for classifier in self.detector.get_active_classifiers():
            try:
                evaluation = classifier.evaluate(X, y)
                results[classifier.name] = evaluation
            except Exception as e:
                results[classifier.name] = {'error': str(e)}
        
        return results
    
    def get_best_model(self, test_data: List[Tuple[Article, bool]]) -> Optional[Classifier]:
        """Get the best performing model"""
        comparison_results = self.compare_models(test_data)
        
        best_model_name = None
        best_f1_score = 0.0
        
        for model_name, results in comparison_results.items():
            if 'f1_score' in results and results['f1_score'] > best_f1_score:
                best_f1_score = results['f1_score']
                best_model_name = model_name
        
        if best_model_name:
            for classifier in self.detector.classifiers:
                if classifier.name == best_model_name:
                    return classifier
        
        return None
