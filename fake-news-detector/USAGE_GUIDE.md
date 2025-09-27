# 🚨 Fake News Detector - Usage Guide

## **🚀 Quick Start**

### **1. Start the Web Application**
```bash
# Navigate to the project directory
cd /Users/srikala/projects/AI-Portfolio/2_fake-news-detector

# Start the web app
uv run python start_app.py
# OR
uv run streamlit run app.py
```

### **2. Test the System**
```bash
# Run the test suite
uv run python test.py

# Test detection functionality
uv run python test_detection.py

# Run the demo
uv run python demo.py
```

## **🌐 Web Interface Usage**

### **Single Article Analysis**
1. **Open the web app** at `http://localhost:8501`
2. **Go to "🔍 Single Article" tab**
3. **Choose input method:**
   - **Text Input**: Enter title, content, and source manually
   - **URL Input**: Provide a URL to analyze (requires web scraping)
   - **File Upload**: Upload a text or CSV file
4. **Click "🚨 Analyze Article"**
5. **View results** with detection status, confidence, and reasoning

### **Batch Analysis**
1. **Go to "📊 Batch Analysis" tab**
2. **Upload a CSV file** with columns: `title`, `content`, `source`, `url`
3. **Click "🚨 Analyze All Articles"**
4. **Download results** as CSV

### **Model Training**
1. **Go to "🤖 Model Training" tab**
2. **Upload training data** (CSV with `is_fake` column)
3. **Click "🚀 Train Models"**
4. **View training results** and model performance

### **Analytics Dashboard**
1. **Go to "📈 Analytics" tab**
2. **View visualizations:**
   - Detection status distribution
   - Confidence score distribution
   - Feature importance charts
3. **Analyze trends** and patterns

## **🔧 Configuration**

### **Detection Settings**
- **Confidence Threshold**: Adjust sensitivity (0.0 - 1.0)
- **Model Selection**: Choose which models to use
- **Feature Extraction**: Enable/disable feature types

### **Model Management**
- **Add Models**: Add new classification models
- **Remove Models**: Remove unwanted models
- **Save/Load**: Persist trained models

## **📊 Understanding Results**

### **Detection Status**
- **🟢 REAL**: Article appears to be legitimate
- **🔴 FAKE**: Article appears to be fake news
- **🟡 UNCERTAIN**: Confidence is too low for definitive classification

### **Confidence Score**
- **0.0 - 0.3**: Low confidence
- **0.3 - 0.7**: Medium confidence
- **0.7 - 1.0**: High confidence

### **Credibility Score**
- **0.0 - 0.3**: Low source credibility
- **0.3 - 0.7**: Medium source credibility
- **0.7 - 1.0**: High source credibility

### **Reasoning**
The system provides human-readable explanations:
- **Sensational language detected**: Title contains alarmist words
- **Excessive punctuation**: Too many exclamation/question marks
- **Low source credibility**: Source is not well-known or trusted
- **High source credibility**: Source is reputable and trusted
- **Very short content**: Article is unusually brief

## **🧪 Testing Examples**

### **Test with Sample Articles**

#### **Real News Example**
```
Title: Scientists Discover New Planet in Habitable Zone
Content: A team of astronomers has discovered a new exoplanet located in the habitable zone of a nearby star. The planet, named Kepler-452b, is approximately 1.6 times the size of Earth and orbits a star similar to our Sun.
Source: NASA
```

**Expected Result**: 🟢 REAL (High credibility, professional language)

#### **Fake News Example**
```
Title: BREAKING: Aliens Contact Earth, Government Hiding Truth!!!
Content: URGENT: Multiple sources confirm that alien spacecraft have been detected near Earth's orbit. Government officials are reportedly covering up the truth from the public.
Source: Conspiracy News Network
```

**Expected Result**: 🔴 FAKE (Sensational language, excessive punctuation)

## **📈 Performance Tips**

### **For Better Accuracy**
1. **Train Models**: Use the Model Training tab with labeled data
2. **Quality Data**: Ensure training data is accurate and diverse
3. **Feature Engineering**: Experiment with different feature combinations
4. **Ensemble Methods**: Use multiple models for better results

### **For Faster Processing**
1. **Batch Processing**: Analyze multiple articles at once
2. **Model Caching**: Save trained models for reuse
3. **Feature Caching**: Cache extracted features
4. **Parallel Processing**: Use multiple CPU cores

## **🔍 Troubleshooting**

### **Common Issues**

#### **"No classifiers available"**
- **Solution**: The system will use basic heuristics
- **Fix**: Train models in the Model Training tab

#### **"spaCy model not found"**
- **Solution**: Semantic features will be limited
- **Fix**: Install spaCy model: `python -m spacy download en_core_web_sm`

#### **"NLTK data not found"**
- **Solution**: Basic stopwords will be used
- **Fix**: Run `python setup_nltk.py`

#### **Low Detection Accuracy**
- **Solution**: Train models with more diverse data
- **Fix**: Use the Model Training tab with quality labeled data

### **Performance Issues**

#### **Slow Processing**
- **Check**: Available memory and CPU
- **Solution**: Use batch processing for multiple articles
- **Fix**: Reduce feature extraction complexity

#### **Memory Errors**
- **Check**: Article content length
- **Solution**: Process articles in smaller batches
- **Fix**: Optimize feature extraction

## **📚 Advanced Usage**

### **Custom Models**
```python
from services import MLClassifier
from models import ModelType

# Add custom classifier
detector.add_classifier(MLClassifier("My Model", ModelType.RANDOM_FOREST))
```

### **Custom Features**
```python
from models import FeatureExtractor

class CustomFeatureExtractor(FeatureExtractor):
    def extract_features(self, text: str) -> Dict[str, float]:
        # Implement custom feature extraction
        return {"custom_feature": 0.5}
```

### **API Integration**
```python
# Use the detector programmatically
result = detector.detect_fake_news(article)
print(f"Detection: {result.is_fake}")
print(f"Confidence: {result.confidence}")
```

## **🎯 Best Practices**

### **Data Preparation**
1. **Clean Data**: Remove HTML, fix encoding issues
2. **Balanced Dataset**: Equal fake/real examples
3. **Quality Labels**: Accurate ground truth labels
4. **Diverse Sources**: Include various news sources

### **Model Training**
1. **Cross-Validation**: Use 5-fold cross-validation
2. **Feature Selection**: Choose relevant features
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Ensemble Methods**: Combine multiple models

### **Evaluation**
1. **Test Set**: Reserve 20% for testing
2. **Metrics**: Use accuracy, precision, recall, F1-score
3. **Confusion Matrix**: Analyze classification errors
4. **ROC Curve**: Evaluate threshold selection

## **🚀 Next Steps**

### **Improvements**
1. **Deep Learning**: Add LSTM/BERT models
2. **Real-time**: Process news feeds in real-time
3. **Multilingual**: Support multiple languages
4. **API Service**: Deploy as a web service

### **Extensions**
1. **Fact Checking**: Integrate with fact-checking APIs
2. **Social Media**: Analyze social media posts
3. **Image Analysis**: Detect fake images
4. **Video Analysis**: Analyze video content

---

**🎉 Happy Fake News Detection!**
