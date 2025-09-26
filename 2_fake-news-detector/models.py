"""
Data models and base classes for the Fake News Detector
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum


class DetectionStatus(Enum):
    """Enumeration of detection status"""
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModelType(Enum):
    """Enumeration of model types"""
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    NAIVE_BAYES = "naive_bayes"
    LSTM = "lstm"
    BERT = "bert"


@dataclass
class Article:
    """Represents a news article with all its metadata"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    author: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate article data after initialization"""
        if not self.title:
            raise ValueError("Article title cannot be empty")
        if not self.content:
            raise ValueError("Article content cannot be empty")
        if not self.source:
            raise ValueError("Article source cannot be empty")
        if not self.url:
            raise ValueError("Article URL cannot be empty")


@dataclass
class DetectionResult:
    """Represents the result of fake news detection"""
    article: Article
    is_fake: bool
    confidence: float
    credibility_score: float
    reasoning: List[str]
    features: Dict[str, float]
    model_predictions: Dict[str, Dict[str, float]]
    detection_status: DetectionStatus
    confidence_level: ConfidenceLevel
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Set derived fields after initialization"""
        if self.confidence >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        else:
            self.confidence_level = ConfidenceLevel.LOW
        
        if self.confidence >= 0.8:
            self.detection_status = DetectionStatus.FAKE if self.is_fake else DetectionStatus.REAL
        else:
            self.detection_status = DetectionStatus.UNCERTAIN


@dataclass
class ModelConfig:
    """Configuration for machine learning models"""
    model_type: ModelType
    parameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    training_data_size: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.model_type:
            raise ValueError("Model type cannot be empty")


class TextProcessor(ABC):
    """Abstract base class for text processing"""
    
    def __init__(self, language: str = "en"):
        self.language = language
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text for analysis"""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        pass
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize text (lemmatization, stemming)"""
        pass
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean text (remove HTML, URLs, etc.)"""
        pass


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract features from text"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass


class Classifier(ABC):
    """Abstract base class for classification models"""
    
    def __init__(self, name: str, model_type: ModelType):
        self.name = name
        self.model_type = model_type
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_names: List[str] = []
    
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict if text is fake news"""
        pass
    
    @abstractmethod
    def train(self, X: List[Dict[str, float]], y: List[bool]) -> None:
        """Train the model"""
        pass
    
    @abstractmethod
    def evaluate(self, X: List[Dict[str, float]], y: List[bool]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save model to file"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load model from file"""
        pass


class CredibilityChecker(ABC):
    """Abstract base class for credibility checking"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def check_credibility(self, article: Article) -> float:
        """Check article credibility (0.0 to 1.0)"""
        pass
    
    @abstractmethod
    def get_credibility_factors(self, article: Article) -> Dict[str, float]:
        """Get detailed credibility factors"""
        pass


class FactChecker(ABC):
    """Abstract base class for fact checking"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def check_facts(self, article: Article) -> Dict[str, Any]:
        """Check facts in article"""
        pass
    
    @abstractmethod
    def get_fact_check_sources(self) -> List[str]:
        """Get available fact-checking sources"""
        pass


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def fetch_articles(self, query: str, limit: int = 10) -> List[Article]:
        """Fetch articles from source"""
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data source"""
        pass


class ModelEvaluator(ABC):
    """Abstract base class for model evaluation"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def evaluate_model(self, classifier: Classifier, 
                      X_test: List[Dict[str, float]], 
                      y_test: List[bool]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def cross_validate(self, classifier: Classifier, 
                      X: List[Dict[str, float]], 
                      y: List[bool], 
                      cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, classifier: Classifier) -> Dict[str, float]:
        """Get feature importance from model"""
        pass
