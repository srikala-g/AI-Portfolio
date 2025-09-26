"""
Concrete implementations of services for the Fake News Detector
"""

import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from datetime import datetime
import requests
from urllib.parse import urlparse
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from textblob import TextBlob
import spacy

from models import (
    TextProcessor, FeatureExtractor, Classifier, CredibilityChecker, 
    FactChecker, DataSource, ModelEvaluator, ModelType, Article
)


class AdvancedTextProcessor(TextProcessor):
    """Advanced text processor with NLP capabilities"""
    
    def __init__(self, language: str = "en"):
        super().__init__(language)
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words(language))
        except:
            # Fallback to basic stopwords if NLTK data is not available
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            try:
                nltk.download('stopwords', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                nltk.download('wordnet', quiet=True)
            except:
                pass
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            try:
                nltk.download('vader_lexicon', quiet=True)
            except:
                pass
    
    def clean(self, text: str) -> str:
        """Clean text by removing HTML, URLs, and special characters"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Clean text
        text = self.clean(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def normalize(self, text: str) -> str:
        """Normalize text using lemmatization"""
        tokens = self.tokenize(text)
        normalized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(normalized_tokens)


class LinguisticFeatureExtractor(FeatureExtractor):
    """Extract linguistic features from text"""
    
    def __init__(self):
        super().__init__("linguistic")
        self.sia = SentimentIntensityAnalyzer()
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features"""
        features = {}
        
        # Basic text statistics
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Readability features
        try:
            features['flesch_score'] = textstat.flesch_reading_ease(text)
            features['fog_index'] = textstat.gunning_fog(text)
            features['smog_index'] = textstat.smog_index(text)
            features['automated_readability'] = textstat.automated_readability_index(text)
        except:
            features['flesch_score'] = 0
            features['fog_index'] = 0
            features['smog_index'] = 0
            features['automated_readability'] = 0
        
        # Sentiment analysis
        try:
            sentiment = TextBlob(text).sentiment
            features['polarity'] = sentiment.polarity
            features['subjectivity'] = sentiment.subjectivity
        except:
            features['polarity'] = 0
            features['subjectivity'] = 0
        
        # VADER sentiment
        try:
            vader_scores = self.sia.polarity_scores(text)
            features['vader_compound'] = vader_scores['compound']
            features['vader_positive'] = vader_scores['pos']
            features['vader_negative'] = vader_scores['neg']
            features['vader_neutral'] = vader_scores['neu']
        except:
            features['vader_compound'] = 0
            features['vader_positive'] = 0
            features['vader_negative'] = 0
            features['vader_neutral'] = 0
        
        # Punctuation and capitalization
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        
        # Word complexity
        words = text.split()
        if words:
            features['long_word_ratio'] = sum(1 for word in words if len(word) > 6) / len(words)
            features['unique_word_ratio'] = len(set(words)) / len(words)
        else:
            features['long_word_ratio'] = 0
            features['unique_word_ratio'] = 0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return [
            'word_count', 'char_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'flesch_score', 'fog_index', 'smog_index',
            'automated_readability', 'polarity', 'subjectivity', 'vader_compound',
            'vader_positive', 'vader_negative', 'vader_neutral', 'exclamation_count',
            'question_count', 'uppercase_ratio', 'digit_ratio', 'long_word_ratio',
            'unique_word_ratio'
        ]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        # This would be populated after model training
        return {name: 0.0 for name in self.get_feature_names()}


class SemanticFeatureExtractor(FeatureExtractor):
    """Extract semantic features from text"""
    
    def __init__(self):
        super().__init__("semantic")
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Semantic features will be limited.")
            self.nlp = None
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract semantic features"""
        features = {}
        
        if not self.nlp:
            return features
        
        try:
            doc = self.nlp(text)
            
            # Named entity features
            features['person_count'] = len([ent for ent in doc.ents if ent.label_ == "PERSON"])
            features['org_count'] = len([ent for ent in doc.ents if ent.label_ == "ORG"])
            features['gpe_count'] = len([ent for ent in doc.ents if ent.label_ == "GPE"])
            features['money_count'] = len([ent for ent in doc.ents if ent.label_ == "MONEY"])
            features['date_count'] = len([ent for ent in doc.ents if ent.label_ == "DATE"])
            
            # POS tag features
            pos_tags = [token.pos_ for token in doc]
            features['noun_ratio'] = pos_tags.count('NOUN') / len(pos_tags) if pos_tags else 0
            features['verb_ratio'] = pos_tags.count('VERB') / len(pos_tags) if pos_tags else 0
            features['adj_ratio'] = pos_tags.count('ADJ') / len(pos_tags) if pos_tags else 0
            features['adv_ratio'] = pos_tags.count('ADV') / len(pos_tags) if pos_tags else 0
            
            # Dependency features
            features['avg_dependency_depth'] = np.mean([token.dep_ for token in doc if token.dep_ != "ROOT"])
            
        except Exception as e:
            print(f"Error in semantic feature extraction: {e}")
            # Return zero features
            features = {
                'person_count': 0, 'org_count': 0, 'gpe_count': 0,
                'money_count': 0, 'date_count': 0, 'noun_ratio': 0,
                'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0,
                'avg_dependency_depth': 0
            }
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return [
            'person_count', 'org_count', 'gpe_count', 'money_count',
            'date_count', 'noun_ratio', 'verb_ratio', 'adj_ratio',
            'adv_ratio', 'avg_dependency_depth'
        ]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return {name: 0.0 for name in self.get_feature_names()}


class MLClassifier(Classifier):
    """Machine Learning classifier implementation"""
    
    def __init__(self, name: str, model_type: ModelType):
        super().__init__(name, model_type)
        self.model = None
        self.feature_names = []
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == ModelType.SVM:
            self.model = SVC(probability=True, random_state=42)
        elif self.model_type == ModelType.LOGISTIC_REGRESSION:
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == ModelType.NAIVE_BAYES:
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict if text is fake news"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to array
        feature_array = np.array([list(features.values())]).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        probability = self.model.predict_proba(feature_array)[0][1]  # Probability of being fake
        
        return bool(prediction), float(probability)
    
    def train(self, X: List[Dict[str, float]], y: List[bool]) -> None:
        """Train the model"""
        if not X or not y:
            raise ValueError("Training data cannot be empty")
        
        # Convert to arrays
        X_array = np.array([list(x.values()) for x in X])
        y_array = np.array(y)
        
        # Store feature names
        self.feature_names = list(X[0].keys())
        
        # Train model
        self.model.fit(X_array, y_array)
        self.is_trained = True
        
        # Calculate accuracy
        predictions = self.model.predict(X_array)
        self.accuracy = accuracy_score(y_array, predictions)
    
    def evaluate(self, X: List[Dict[str, float]], y: List[bool]) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_array = np.array([list(x.values()) for x in X])
        y_array = np.array(y)
        
        predictions = self.model.predict(X_array)
        
        return {
            'accuracy': accuracy_score(y_array, predictions),
            'precision': precision_score(y_array, predictions, average='weighted'),
            'recall': recall_score(y_array, predictions, average='weighted'),
            'f1_score': f1_score(y_array, predictions, average='weighted')
        }
    
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'accuracy': self.accuracy
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.accuracy = model_data['accuracy']
        self.is_trained = True


class SourceCredibilityChecker(CredibilityChecker):
    """Check credibility based on source information"""
    
    def __init__(self):
        super().__init__("source_credibility")
        # Known credible and non-credible domains
        self.credible_domains = {
            'bbc.com', 'reuters.com', 'ap.org', 'npr.org', 'pbs.org',
            'nytimes.com', 'washingtonpost.com', 'wsj.com', 'ft.com',
            'economist.com', 'guardian.com', 'independent.co.uk'
        }
        
        self.non_credible_domains = {
            'infowars.com', 'breitbart.com', 'naturalnews.com',
            'beforeitsnews.com', 'yournewswire.com'
        }
    
    def check_credibility(self, article: Article) -> float:
        """Check article credibility based on source"""
        domain = urlparse(article.url).netloc.lower()
        
        # Check against known domains
        if domain in self.credible_domains:
            return 0.9  # High credibility
        elif domain in self.non_credible_domains:
            return 0.1  # Low credibility
        else:
            # Unknown domain - moderate credibility
            return 0.5
    
    def get_credibility_factors(self, article: Article) -> Dict[str, float]:
        """Get detailed credibility factors"""
        domain = urlparse(article.url).netloc.lower()
        
        factors = {
            'domain_reputation': 0.5,
            'ssl_certificate': 0.5,
            'domain_age': 0.5,
            'social_media_presence': 0.5
        }
        
        # Check domain reputation
        if domain in self.credible_domains:
            factors['domain_reputation'] = 0.9
        elif domain in self.non_credible_domains:
            factors['domain_reputation'] = 0.1
        
        return factors


class APIFactChecker(FactChecker):
    """Fact checker using external APIs"""
    
    def __init__(self, api_key: str = None):
        super().__init__("api_fact_checker")
        self.api_key = api_key
        self.fact_check_sources = [
            'factcheck.org', 'snopes.com', 'politifact.com',
            'factchecker.in', 'boomlive.in'
        ]
    
    def check_facts(self, article: Article) -> Dict[str, Any]:
        """Check facts in article using external APIs"""
        result = {
            'fact_check_available': False,
            'fact_check_result': None,
            'sources_checked': [],
            'confidence': 0.0
        }
        
        # This would integrate with fact-checking APIs
        # For now, return a mock result
        if self.api_key:
            result['fact_check_available'] = True
            result['confidence'] = 0.7
            result['sources_checked'] = self.fact_check_sources[:2]
        
        return result
    
    def get_fact_check_sources(self) -> List[str]:
        """Get available fact-checking sources"""
        return self.fact_check_sources


class NewsAPIDataSource(DataSource):
    """Data source using NewsAPI"""
    
    def __init__(self, api_key: str):
        super().__init__("newsapi")
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def fetch_articles(self, query: str, limit: int = 10) -> List[Article]:
        """Fetch articles from NewsAPI"""
        articles = []
        
        try:
            url = f"{self.base_url}/everything"
            params = {
                "q": query,
                "apiKey": self.api_key,
                "pageSize": limit,
                "sortBy": "publishedAt",
                "language": "en"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for article_data in data.get("articles", []):
                article = Article(
                    title=article_data.get("title", ""),
                    content=article_data.get("content", ""),
                    source=article_data.get("source", {}).get("name", ""),
                    url=article_data.get("url", ""),
                    published_at=datetime.fromisoformat(
                        article_data.get("publishedAt", "").replace("Z", "+00:00")
                    ),
                    author=article_data.get("author", "")
                )
                articles.append(article)
        
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
        
        return articles
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source"""
        return {
            "name": "NewsAPI",
            "description": "Real-time news from 80,000+ sources",
            "api_documentation": "https://newsapi.org/docs",
            "rate_limit": "1000 requests per day (free tier)"
        }


class ModelEvaluatorImpl(ModelEvaluator):
    """Model evaluator implementation"""
    
    def __init__(self):
        super().__init__("model_evaluator")
    
    def evaluate_model(self, classifier: Classifier, 
                      X_test: List[Dict[str, float]], 
                      y_test: List[bool]) -> Dict[str, float]:
        """Evaluate model performance"""
        return classifier.evaluate(X_test, y_test)
    
    def cross_validate(self, classifier: Classifier, 
                      X: List[Dict[str, float]], 
                      y: List[bool], 
                      cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        if not classifier.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        X_array = np.array([list(x.values()) for x in X])
        y_array = np.array(y)
        
        # Perform cross-validation
        cv_scores = cross_val_score(classifier.model, X_array, y_array, cv=cv_folds)
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def get_feature_importance(self, classifier: Classifier) -> Dict[str, float]:
        """Get feature importance from model"""
        if not classifier.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(classifier.model, 'feature_importances_'):
            importance_dict = dict(zip(classifier.feature_names, classifier.model.feature_importances_))
            return importance_dict
        else:
            return {name: 0.0 for name in classifier.feature_names}
