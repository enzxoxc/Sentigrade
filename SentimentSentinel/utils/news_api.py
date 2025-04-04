import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import time
import google.generativeai as genai
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

def setup_api_keys():
    """
    Setup API keys for Google Custom Search and Gemini API.
    Uses hardcoded API keys as provided.
    
    Returns:
        tuple: (API_KEY, CSE_ID, GEMINI_API_KEY, api_configured)
    """
    # Use the provided API keys
    api_key = "AIzaSyDnurZ5JrfyX0lo6LFULMP6UDfERoMfafo"
    cse_id = "274d86c6927de4b39"
    gemini_api_key = api_key  # Use the same key for Gemini
    
    api_configured = True  # Always configured with hardcoded keys
    
    return api_key, cse_id, gemini_api_key, api_configured

def search_news(query: str, api_key: Optional[str], cse_id: Optional[str], max_results: int = 10, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, str]]:
    if not api_key or not cse_id:
        st.error("API Key or CSE ID not configured.")
        return []
    
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        
        keywords = [k.strip() for k in query.split(",")]
        refined_query = f'"{" AND ".join(keywords)}" site:news'
        
        if start_date and end_date:
            refined_query += f" after:{start_date} before:{end_date}"

        start_index = 1
        results = []
        while len(results) < max_results:
            res = service.cse().list(q=refined_query, cx=cse_id, num=10, start=start_index).execute()
            if 'items' in res:
                results.extend(res['items'])
                start_index += 10
            else:
                break
            
        news_articles = []
        for item in results[:max_results]:
            if any(keyword.lower() in item['title'].lower() for keyword in keywords):
                published_date = item.get('publishedTime', None)
                if published_date:
                    try:
                        # Parsing the date in the format returned by the API
                        published_date = datetime.strptime(published_date, "%a, %d %b %Y %H:%M:%S %Z")
                    except ValueError:
                        published_date = None
                news_articles.append({
                    'title': item['title'],
                    'link': item['link'],
                    'snippet': item.get('snippet', 'No snippet available'),
                    'source': item.get('displayLink', 'Unknown source'),
                    'date': published_date if published_date else 'Unknown date'
                })
        
        # Sort the articles by date in descending order
        news_articles.sort(key=lambda x: x['date'], reverse=True)
        
        return news_articles[:max_results]  # Return the top `max_results` most recent articles
    
    except Exception as e:
        st.error(f"Error searching news: {str(e)}")
        return []

def gemini_analyze_sentiment(text: str, api_key: Optional[str]) -> float:
    """
    Analyze sentiment of text using Google Gemini API.
    
    Args:
        text (str): Text to analyze sentiment for
        api_key (Optional[str]): Gemini API Key
        
    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive)
    """
    if not api_key:
        st.error("Gemini API Key not configured.")
        return 0.0
    
    try:
        # Configure the API key exactly as in the notebook
        genai.configure(api_key=api_key)
        
        # Create the model with the exact same model name
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Call generate_content directly as shown in the notebook
        response = model.generate_content(
            f"Please analyze the sentiment of the following headline and return only the sentiment score between -10 (negative) and 10 (positive), no explanations or reasons: {text}."
        )
        
        # Extract the sentiment score exactly as in the notebook
        try:
            sentiment_score = float(response.text.strip())
            
            # Ensure the score is in the range [-10, 10]
            sentiment_score = max(-10, min(10, sentiment_score))
            
            # Normalize to range [-1, 1] for consistency with our app
            normalized_score = sentiment_score / 10
            
            return normalized_score
        except ValueError:
            st.warning(f"Could not convert sentiment response to number: {response.text}")
            return 0.0
    except Exception as e:
        st.error(f"Error with Gemini API: {str(e)}")
        return 0.0

def fetch_and_analyze_news(queries: List[str], 
                          max_results_per_query: int = 10,
                          with_progress: bool = True) -> pd.DataFrame:
    """
    Fetch news for multiple queries and analyze sentiment.
    
    Args:
        queries (List[str]): List of search queries
        max_results_per_query (int): Maximum results per query
        with_progress (bool): Whether to show a progress bar
        
    Returns:
        pd.DataFrame: DataFrame with news and sentiment data
    """
    # Setup API keys
    api_key, cse_id, gemini_api_key, api_configured = setup_api_keys()
    
    if not api_configured:
        st.error("API keys not configured. Please set GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables.")
        return pd.DataFrame()
    
    # Calculate the date range (7 days before today)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=7)
    
    # Display the date range
    st.write(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Show progress if requested
    if with_progress:
        progress_bar = st.progress(0)
        progress_text = st.empty()
    
    all_news = []
    total_articles_analyzed = 0
    progress_bar = None
    progress_text = None
    
    # Process each query
    for i, query in enumerate(queries):
        if with_progress and progress_text is not None:
            progress_text.text(f"Searching for news related to: {query}")
        
        # Search for news
        if api_key is not None and cse_id is not None:
            news_articles = search_news(query, api_key, cse_id, max_results_per_query, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))
            
            # Process each article
            for article in news_articles:
                if with_progress and progress_text is not None:
                    progress_text.text(f"Analyzing sentiment for: {article['title']}")
                
                # Analyze sentiment
                sentiment_score = 0.0
                if gemini_api_key is not None:
                    sentiment_score = gemini_analyze_sentiment(article['title'], gemini_api_key)
                
                # Add to results
                all_news.append({
                    'query': query,
                    'title': article['title'],
                    'link': article['link'],
                    'snippet': article['snippet'],
                    'source': article['source'],
                    'date': article['date'],
                    'sentiment_score': sentiment_score
                })
                
                # Count analyzed articles
                total_articles_analyzed += 1
                
                # Add a small delay to avoid hitting API rate limits
                time.sleep(random.uniform(0.5, 1.5))
        
        # Update progress
        if with_progress and progress_bar is not None and len(queries) > 0:
            progress_bar.progress((i + 1) / len(queries))
    
    # Clear progress indicators
    if with_progress:
        if progress_bar is not None:
            progress_bar.empty()
        if progress_text is not None:
            progress_text.empty()
    
    # Create DataFrame
    if all_news:
        df = pd.DataFrame(all_news)
        # Store the total number of articles analyzed in the session state
        st.session_state.total_articles_analyzed = total_articles_analyzed
        return df
    else:
        st.session_state.total_articles_analyzed = 0
        return pd.DataFrame()

def categorize_sentiment(score: float) -> str:
    """
    Categorize sentiment score into positive, neutral, or negative.
    
    Args:
        score (float): Sentiment score between -1 and 1
        
    Returns:
        str: Sentiment category ('positive', 'neutral', or 'negative')
    """
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"
