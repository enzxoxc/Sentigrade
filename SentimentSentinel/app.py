import streamlit as st
from utils.news_api import fetch_and_analyze_news, setup_api_keys, categorize_sentiment
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration with a professional, clean look
st.set_page_config(
    page_title="Sentigrade - News Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to improve the look and feel with a pure white, ultra-clean Google-like style
st.markdown("""
<style>
    body {
        font-family: 'Roboto', Arial, sans-serif;
        color: #202124;
        background-color: #ffffff;
    }
    
    .main {
        padding-top: 2rem;
        background-color: #ffffff;
    }
    
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #ffffff;
    }
    
    h1, h2, h3 {
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        font-weight: 400;
        color: #202124;
        letter-spacing: -0.5px;
    }
    
    p {
        font-family: 'Roboto', Arial, sans-serif;
        color: #3c4043;
        line-height: 1.5;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #1a73e8;
        color: white;
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        font-weight: 500;
        border: none;
        transition: background-color 0.2s;
        box-shadow: none;
    }
    
    .stButton>button:hover {
        background-color: #185abc;
        box-shadow: 0 1px 3px rgba(60,64,67,0.3);
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .css-18e3th9, .css-1d391kg, .css-12oz5g7 {
        padding-top: 0;
        background-color: #ffffff;
    }
    
    .metric-container {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(60,64,67,0.1);
        padding: 24px;
        margin-bottom: 24px;
        text-align: center;
        border: 1px solid #f1f3f4;
    }
    
    .news-card {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(60,64,67,0.1);
        padding: 20px;
        margin-bottom: 16px;
        transition: box-shadow 0.2s;
        border: 1px solid #f1f3f4;
    }
    
    .news-card:hover {
        box-shadow: 0 1px 3px rgba(60,64,67,0.2);
    }
    
    .sentiment-badge {
        border-radius: 16px;
        padding: 6px 12px;
        font-weight: 500;
        display: inline-block;
        text-align: center;
        margin-left: 10px;
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        font-size: 13px;
    }
    
    .positive {
        background-color: rgba(52, 168, 83, 0.1);
        color: #34a853;
        border: 1px solid rgba(52, 168, 83, 0.2);
    }
    
    .neutral {
        background-color: rgba(251, 188, 5, 0.1);
        color: #f29900;
        border: 1px solid rgba(251, 188, 5, 0.2);
    }
    
    .negative {
        background-color: rgba(234, 67, 53, 0.1);
        color: #ea4335;
        border: 1px solid rgba(234, 67, 53, 0.2);
    }
    
    .dashboard-stats {
        display: flex;
        justify-content: space-between;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(60,64,67,0.1);
        padding: 16px;
        text-align: center;
        flex: 1;
        margin: 0 8px;
        border: 1px solid #f1f3f4;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: 500;
        color: #1a73e8;
        font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
        margin-bottom: 4px;
    }
    
    .stat-label {
        font-size: 14px;
        color: #5f6368;
        font-family: 'Roboto', Arial, sans-serif;
    }
    
    /* Google-like clean input field */
    .stTextInput>div>div>input {
        border-radius: 24px;
        border: 1px solid #dfe1e5;
        padding: 12px 16px;
        font-size: 16px;
        color: #202124;
        background-color: #ffffff;
        font-family: 'Roboto', Arial, sans-serif;
        box-shadow: none;
    }
    
    .stTextInput>div>div>input:focus {
        border: 1px solid #1a73e8;
        box-shadow: 0 1px 2px rgba(60,64,67,0.1);
    }
    
    /* Clean containers */
    .element-container, .stPlotlyChart {
        background-color: #ffffff;
    }
    
    /* All Streamlit containers should be white */
    .css-1y4p8pa, .css-1r6slb0, .css-12oz5g7, .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
        background-color: #ffffff;
    }
    
    /* Override any dark elements */
    .css-1e5imcs, .css-14xtw13 {
        border-color: #f1f3f4;
        color: #3c4043;
    }
    
    /* Make plot backgrounds white */
    .js-plotly-plot .plotly .main-svg {
        background-color: #ffffff !important;
    }
    
    /* Cleaner select box */
    .stSelectbox>div>div>div {
        background-color: #ffffff;
        border-radius: 24px;
        border: 1px solid #dfe1e5;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing data
if "news_data" not in st.session_state:
    st.session_state.news_data = None
    
if "overall_sentiment" not in st.session_state:
    st.session_state.overall_sentiment = None
    
if "total_articles_analyzed" not in st.session_state:
    st.session_state.total_articles_analyzed = 0
    
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Application header with modern design - Google style
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #1a73e8; margin-bottom: 0.5rem; font-family: 'Google Sans', 'Roboto', Arial, sans-serif; font-weight: 400;">Sentigrade</h1>
    <p style="font-size: 1.2rem; color: #5f6368; font-family: 'Roboto', Arial, sans-serif;">AI-powered sentiment analysis for Southeast Asian news</p>
</div>
""", unsafe_allow_html=True)

# Clean, modern interface with clear search box - Google style
st.markdown("""
<div style="max-width: 800px; margin: 0 auto; background-color: #ffffff; padding: 2rem; border-radius: 8px; border: 1px solid #f1f3f4;">
    <h3 style="margin-bottom: 1rem; color: #202124; font-family: 'Google Sans', 'Roboto', Arial, sans-serif; font-weight: 400;">Analyze News Sentiment</h3>
    <p style="color: #5f6368; margin-bottom: 1.5rem; font-family: 'Roboto', Arial, sans-serif;">Enter keywords, topics, or phrases to analyze sentiment in news articles across Southeast Asia.</p>
</div>
""", unsafe_allow_html=True)

# Create a clean search interface
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input(
        "Search query",
        value=st.session_state.get('search_query', ''),
        placeholder="Try 'Singapore business', 'Malaysia tech', 'Thailand tourism'...",
        label_visibility="collapsed"
    )
with col2:
    run_analysis = st.button("✨ Analyze", use_container_width=True)

# Run analysis when button is clicked
if run_analysis:
    if search_query:
        # Save the search query to session state for persistence
        st.session_state.search_query = search_query
        
        # Add to search history
        if search_query not in st.session_state.search_history:
            st.session_state.search_history.append(search_query)
            # Keep only the last 5 searches
            if len(st.session_state.search_history) > 5:
                st.session_state.search_history = st.session_state.search_history[-5:]
        
        with st.spinner("Analyzing sentiment in news articles..."):
            # Parse the search terms
            search_terms = [search_query.strip()]
            
            # Fetch news data
            news_df = fetch_and_analyze_news(
                queries=search_terms,
                max_results_per_query=7,  # Increased for better analysis
                with_progress=True  # Show progress
            )
            
            # Store in session state
            st.session_state.news_data = news_df
            
            # Calculate average sentiment
            if not news_df.empty:
                st.session_state.overall_sentiment = news_df['sentiment_score'].mean()
    else:
        st.warning("Please enter keywords to analyze.")

# Format sentiment scores for display
def format_sentiment(sentiment_value):
    if sentiment_value is None:
        return "N/A"
    else:
        # Format to 2 decimal places and add + sign for positive values
        formatted = f"{sentiment_value:.2f}"
        if sentiment_value > 0:
            formatted = "+" + formatted
        return formatted

# Dashboard overview section
if st.session_state.news_data is not None and not st.session_state.news_data.empty:
    # Dashboard stats
    total_articles = len(st.session_state.news_data)
    positive_articles = len(st.session_state.news_data[st.session_state.news_data['sentiment_score'] > 0.05])
    negative_articles = len(st.session_state.news_data[st.session_state.news_data['sentiment_score'] < -0.05])
    neutral_articles = total_articles - positive_articles - negative_articles
    
    # Overall sentiment metric
    overall_sentiment = st.session_state.overall_sentiment
    
    # Determine sentiment label and color using Google's palette
    if overall_sentiment is None:
        sentiment_label = "Unknown"
        sentiment_color = "#9AA0A6"  # Google grey
    elif overall_sentiment > 0.05:
        sentiment_label = "Positive"
        sentiment_color = "#34A853"  # Google green
    elif overall_sentiment < -0.05:
        sentiment_label = "Negative"
        sentiment_color = "#EA4335"  # Google red
    else:
        sentiment_label = "Neutral"
        sentiment_color = "#FBBC05"  # Google yellow
    
    # Dashboard layout
    st.markdown("<h2 style='text-align: center; margin-top: 2rem; font-family: \"Google Sans\", \"Roboto\", Arial, sans-serif; font-weight: 400; color: #202124;'>Sentiment Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    # Show analysis statistics
    st.markdown("""
    <div class="dashboard-stats">
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Articles Analyzed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Positive Articles</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Neutral Articles</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Negative Articles</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{}</div>
            <div class="stat-label">Queries Completed</div>
        </div>
    </div>
    """.format(
        total_articles, 
        positive_articles,
        neutral_articles,
        negative_articles,
        st.session_state.total_articles_analyzed
    ), unsafe_allow_html=True)
    
    # Display overall sentiment score in a modern metric box
    st.markdown(f"""
    <div class="metric-container">
        <h3>Overall Sentiment</h3>
        <div style="font-size: 48px; font-weight: bold; color: {sentiment_color}; margin: 10px 0;">
            {format_sentiment(overall_sentiment)}
        </div>
        <div style="font-size: 24px; color: {sentiment_color}; font-weight: 500;">
            {sentiment_label}
        </div>
        <div style="font-size: 14px; color: #666; margin-top: 10px;">
            Based on {total_articles} articles related to "{st.session_state.search_query}"
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add sentiment category for visualization
    news_df = st.session_state.news_data.copy()
    news_df['sentiment_category'] = news_df['sentiment_score'].apply(categorize_sentiment)
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a professional gauge chart for sentiment
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overall_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Gauge", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': sentiment_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.05], 'color': 'rgba(234, 67, 53, 0.3)'},  # Google red
                    {'range': [-0.05, 0.05], 'color': 'rgba(251, 188, 5, 0.3)'}, # Google yellow
                    {'range': [0.05, 1], 'color': 'rgba(52, 168, 83, 0.3)'}      # Google green
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': overall_sentiment
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={"color": "#1f77b4", "family": "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution chart - upgraded pie chart
        sentiment_counts = news_df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Set a custom color scale for sentiment categories using Google colors
        color_map = {'positive': '#34A853', 'neutral': '#FBBC05', 'negative': '#EA4335'}
        
        fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            title="<b>Sentiment Distribution</b>",
            color='Sentiment',
            color_discrete_map=color_map,
            hole=0.4,
        )
        
        fig.update_traces(
            textinfo='percent+label', 
            textfont_size=14,
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        
        fig.update_layout(
            legend_title_text='',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            height=350,
            margin=dict(l=20, r=20, t=50, b=80),
            title_x=0.5,
            paper_bgcolor="white",
            font={"family": "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced horizontal bar chart
        news_df['abs_sentiment'] = news_df['sentiment_score'].abs()
        
        # Sort by absolute sentiment for better visualization
        sorted_df = news_df.sort_values('abs_sentiment', ascending=False).head(8)
        
        # Truncate long titles
        sorted_df['short_title'] = sorted_df['title'].apply(lambda x: (x[:40] + '...') if len(x) > 40 else x)
        
        # Create a custom color scale based on sentiment values
        fig = px.bar(
            sorted_df, 
            x='sentiment_score', 
            y='short_title', 
            orientation='h',
            title="<b>Top Headlines by Sentiment Impact</b>",
            color='sentiment_score',
            color_continuous_scale=['#EA4335', '#FBBC05', '#34A853'],  # Google colors
            labels={'sentiment_score': 'Sentiment Score', 'short_title': 'Headline'}
        )
        
        # Add a vertical line at x=0 to indicate neutral sentiment
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # Format the y-axis labels to truncate long headlines
        fig.update_traces(hovertemplate='<b>%{customdata}</b><br>Sentiment: %{x:.2f}', customdata=sorted_df['title'])
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            title_x=0.5,
            xaxis_title="Sentiment Score",
            yaxis_title="",
            coloraxis_colorbar=dict(
                title="Sentiment",
                tickvals=[-1, 0, 1],
                ticktext=["Negative", "Neutral", "Positive"],
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Source distribution
        source_sentiment = news_df.groupby('source')['sentiment_score'].mean().reset_index()
        source_sentiment = source_sentiment.sort_values('sentiment_score', ascending=False)
        
        fig = px.bar(
            source_sentiment,
            x='source',
            y='sentiment_score',
            title="<b>Average Sentiment by News Source</b>",
            color='sentiment_score',
            color_continuous_scale=['#EA4335', '#FBBC05', '#34A853'],  # Google colors
            labels={'sentiment_score': 'Avg. Sentiment', 'source': 'News Source'}
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=80),
            title_x=0.5,
            xaxis_title="",
            yaxis_title="Average Sentiment",
            xaxis_tickangle=-45,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # News articles section with improved cards
    st.markdown("<h3 style='margin-top: 2rem; font-family: \"Google Sans\", \"Roboto\", Arial, sans-serif; font-weight: 400; color: #202124;'>Top News Articles</h3>", unsafe_allow_html=True)
    
    # Get headlines with the strongest sentiment (both positive and negative)
    top_headlines = news_df.sort_values('abs_sentiment', ascending=False)
    
    # Display as modern cards with improved formatting
    for _, row in top_headlines.iterrows():
        sentiment = row['sentiment_category']
        score = row['sentiment_score']
        
        st.markdown(f"""
        <div class="news-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 1;">
                    <h4 style="margin-top: 0; font-family: 'Google Sans', 'Roboto', Arial, sans-serif; font-weight: 400; color: #202124;">
                        <a href="{row['link']}" target="_blank" style="color: #1a73e8; text-decoration: none;">
                            {row['title']}
                        </a>
                    </h4>
                    <p style="color: #5f6368; margin-bottom: 8px; font-size: 14px; font-family: 'Roboto', Arial, sans-serif;">
                        <span style="font-weight: 500;">Source:</span> {row['source']} | 
                        <span style="font-weight: 500;">Query:</span> {row['query']}
                    </p>
                    <p style="margin-bottom: 0; color: #3c4043; font-family: 'Roboto', Arial, sans-serif; line-height: 1.5;">
                        {row['snippet']}
                    </p>
                </div>
                <div style="margin-left: 15px;">
                    <span class="sentiment-badge {sentiment}">{sentiment.upper()}<br>{score:.2f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Show empty state with improved UI
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; background-color: #ffffff; border-radius: 8px; margin: 2rem 0; border: 1px solid #f1f3f4;">
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#1a73e8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <h3 style="margin-top: 1rem; color: #202124; font-family: 'Google Sans', 'Roboto', Arial, sans-serif; font-weight: 400;">Ready to analyze news sentiment</h3>
        <p style="color: #5f6368; max-width: 500px; margin: 1rem auto; font-family: 'Roboto', Arial, sans-serif;">
            Enter keywords in the search box above and click 'Analyze' to see sentiment analysis results from news sources across Southeast Asia.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggest recent searches if available
    if st.session_state.search_history:
        st.markdown("<h4 style='font-family: \"Google Sans\", \"Roboto\", Arial, sans-serif; font-weight: 400; color: #202124; margin-top: 2rem;'>Recent searches:</h4>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, query in enumerate(reversed(st.session_state.search_history)):
            with cols[i % 5]:
                if st.button(f"🔍 {query}", key=f"recent_{i}"):
                    st.session_state.search_query = query
                    st.rerun()
    
    # Example search suggestions with improved UI
    st.markdown("<h4 style='font-family: \"Google Sans\", \"Roboto\", Arial, sans-serif; font-weight: 400; color: #202124; margin-top: 1.5rem;'>Try these examples:</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Singapore business"):
            st.session_state.search_query = "Singapore business"
            st.rerun()
    
    with col2:
        if st.button("Malaysia politics"):
            st.session_state.search_query = "Malaysia politics"
            st.rerun()
    
    with col3:
        if st.button("Thailand tourism"):
            st.session_state.search_query = "Thailand tourism"
            st.rerun()

# Modern Google-style footer
st.markdown("""
<div style="text-align: center; padding: 1rem 0; margin-top: 3rem; border-top: 1px solid #f1f3f4;">
    <p style="color: #5f6368; font-size: 14px; font-family: 'Roboto', Arial, sans-serif;">
        © 2025 Sentigrade • AI-powered news sentiment analysis for Southeast Asia
    </p>
</div>
""", unsafe_allow_html=True)
