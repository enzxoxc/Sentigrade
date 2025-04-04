import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Initialize NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text, language_code="en"):
    """
    Analyze sentiment of text using NLTK's VADER SentimentIntensityAnalyzer.
    Used as a fallback when Gemini API is not functioning.
    
    Args:
        text (str): Text to analyze
        language_code (str): Language code for the text (default: English)
        
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text:
        return 0.0
    
    # Clean the text
    text = clean_text(text)
    
    # Use VADER for sentiment analysis
    scores = sia.polarity_scores(text)
    
    # Return the compound score which is in range [-1, 1]
    return scores['compound']
        
def clean_text(text):
    """
    Clean text for sentiment analysis.
    
    Args:
        text (str): Text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions 
    text = re.sub(r'@\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
