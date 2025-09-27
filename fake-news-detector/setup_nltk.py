#!/usr/bin/env python3
"""
Setup script to download required NLTK data
"""

import nltk
import ssl

def download_nltk_data():
    """Download required NLTK data"""
    print("🔄 Downloading NLTK data...")
    
    # Handle SSL certificate issues
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required NLTK data
    nltk_data = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words'
    ]
    
    for data in nltk_data:
        try:
            print(f"📥 Downloading {data}...")
            nltk.download(data, quiet=True)
            print(f"✅ {data} downloaded successfully")
        except Exception as e:
            print(f"❌ Error downloading {data}: {e}")
    
    print("🎉 NLTK data download completed!")

if __name__ == "__main__":
    download_nltk_data()
