import streamlit as st
import requests
import json
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

# Set page configuration
st.set_page_config(
    page_title="AI Mental Health Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint
API_URL = "http://localhost:5000/api"

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_score' not in st.session_state:
    st.session_state.current_score = None

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4F8BF9; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #4F8BF9; margin-bottom: 1rem; }
    .score-container { padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; }
    .normal { background-color: #ABEBC6; color: #196F3D; }
    .stress { background-color: #FCF3CF; color: #7D6608; }
    .anxiety { background-color: #FADBD8; color: #78281F; }
    .depression { background-color: #F5B7B1; color: #7B241C; }
    .suicidal { background-color: #EC7063; color: white; }
    
    .ml-prediction {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #6c5ce7;
        margin-bottom: 1.5rem;
    }
    .ml-class {
        font-size: 1.8rem;
        font-weight: bold;
        color: #6c5ce7;
    }
    
    .suggestion-card {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        background-color: #EBF5FB;
        border-left: 5px solid #3498DB;
    }
    .suggestion-category { font-weight: bold; color: #3498DB; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<p class="sub-header">AI Mental Health Monitor</p>', unsafe_allow_html=True)
    st.image("https://img.icons8.com/color/96/000000/mental-health.png", width=100)
    
    st.markdown("### User ID")
    st.code(st.session_state.user_id, language="text")
    
    st.markdown("### Navigation")
    page = st.radio("Go to", ["Home", "Text Analysis", "History", "About"])
    
    st.markdown("---")
    st.markdown("### Privacy Notice")
    st.info("Your data is processed locally and securely. You maintain full control over your information at all times.")

def call_text_api(text):
    """Call the text analysis API"""
    try:
        response = requests.post(
            f"{API_URL}/analyze/text",
            json={
                "text": text,
                "user_id": st.session_state.user_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned an error: {response.text}")
            return None
    except requests.RequestException as e:
        st.error(f"Failed to connect to the backend API: {str(e)}")
        st.error("Please ensure the Flask backend is running on port 5000.")
        return None

def display_score(score_data):
    """Display the mental health score and analysis"""
    if not score_data:
        return
    
    score = score_data.get("score", 50)
    category = score_data.get("category", "moderate")
    
    # Store the current score
    st.session_state.current_score = score_data
    
    # Add to history if not already there
    timestamp = datetime.datetime.now().isoformat()
    history_item = {"timestamp": timestamp, "score": score, "category": category}
    st.session_state.history.append(history_item)
    
    # Extract ML Prediction
    analysis = score_data.get("analysis", {})
    ml_prediction = analysis.get("ml_prediction", None)
    
    if ml_prediction:
        predicted_class = ml_prediction.get("class", "Unknown")
        probabilities = ml_prediction.get("probabilities", {})
        
        st.markdown(f"<div class='ml-prediction'>" +
                    f"<h3>Machine Learning Classification</h3>" +
                    f"<div class='ml-class'>{predicted_class}</div>" +
                    f"<p>Our trained Logistic Regression model has analyzed your text and determined this is the most likely condition.</p>" +
                    f"</div>", unsafe_allow_html=True)
        
        st.markdown("#### Class Probabilities")
        cols = st.columns(len(probabilities))
        for i, (cls_name, prob) in enumerate(probabilities.items()):
            with cols[i]:
                st.metric(label=cls_name, value=f"{prob*100:.1f}%")
                st.progress(max(0.0, min(1.0, float(prob))))
        st.markdown("---")

    # Display Overall System Score
    category_css = category.lower()
    st.markdown(f"<div class='score-container {category_css}'>" +
                f"<h2>System Wellness Score: {score:.1f}/100</h2>" +
                f"<h3>Identified Category: {category}</h3>" +
                f"</div>", unsafe_allow_html=True)
    
    # Display suggestions
    st.markdown("### Personalized Suggestions")
    for suggestion in score_data.get("suggestions", []):
        st.markdown(
            f"<div class='suggestion-card'>" +
            f"<span class='suggestion-category'>{suggestion['category'].capitalize()}</span>: {suggestion['text']}" +
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Display detailed NLP analysis
    with st.expander("View NLP Text Metrics"):
        if "sentiment" in analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**VADER Sentiment Scores**")
                sentiment = analysis["sentiment"]
                st.progress(sentiment.get("pos", 0), "Positive")
                st.progress(sentiment.get("neu", 0), "Neutral")
                st.progress(sentiment.get("neg", 0), "Negative")
                
                st.markdown("**Text Complexity Metrics**")
                metrics = analysis.get("metrics", {})
                st.metric("Word Count", metrics.get("word_count", 0))
                st.metric("Linguistic Diversity", f"{metrics.get('linguistic_diversity', 0):.2f}")
            
            with col2:
                st.markdown("**Rule-Based Linguistic Markers**")
                st.caption("Matches against predefined keyword dictionaries")
                markers = analysis.get("markers", {})
                st.metric("Depression Keyword Matches", markers.get("depression", 0))
                st.metric("Anxiety Keyword Matches", markers.get("anxiety", 0))
                st.metric("Stress Keyword Matches", markers.get("stress", 0))

# Main content based on selected page
if page == "Home":
    st.markdown('<p class="main-header">AI Mental Health Monitor</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the AI Mental Health Monitor! This application uses a custom-trained Machine Learning model 
    to analyze your text entries for early signs of mental health concerns.
    
    ### How It Works
    1. **Text Analysis**: Enter your thoughts or journal entries.
    2. **Machine Learning Model**: Our Logistic Regression AI classifies your text into categories like Anxiety, Depression, Stress, or Normal.
    3. **NLP Metrics**: The system calculates a holistic wellness score using VADER sentiment analysis and linguistic markers.
    4. **Track Progress**: View your mental health trends over time.
    """)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("📝 Start Text Analysis", use_container_width=True):
            st.session_state.page = "Text Analysis"
            st.experimental_rerun()
    
    # Show recent history if available
    if st.session_state.history:
        st.markdown("### Recent Activity")
        recent_history = st.session_state.history[-5:]
        
        if len(recent_history) > 1:
            data = pd.DataFrame(recent_history)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp')
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['timestamp'], data['score'], marker='o', linestyle='-', color='#4F8BF9')
            ax.set_ylim(0, 100)
            ax.set_ylabel('System Wellness Score')
            ax.set_xlabel('Date')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig)

elif page == "Text Analysis":
    st.markdown('<p class="main-header">Text Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Enter your thoughts, feelings, or a journal entry below. The AI will classify your text 
    using its trained ML model and provide NLP metrics.
    """)
    
    text_input = st.text_area("Share your thoughts...", height=200)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Analyze Text", use_container_width=True):
            if text_input.strip():
                with st.spinner("Running Machine Learning model..."):
                    result = call_text_api(text_input)
                    if result:
                        display_score(result)
            else:
                st.warning("Please enter some text to analyze.")
    
    if st.session_state.current_score:
        display_score(st.session_state.current_score)

elif page == "History":
    st.markdown('<p class="main-header">Your History</p>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No history available yet. Complete a text analysis to see your results here.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        if len(history_df) > 1:
            st.markdown("### Wellness Score Trend")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history_df['timestamp'], history_df['score'], marker='o', linestyle='-', color='#4F8BF9')
            ax.set_ylim(0, 100)
            ax.set_ylabel('Wellness Score')
            ax.set_xlabel('Date')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.axhspan(80, 100, alpha=0.2, color='green', label='Normal')
            ax.axhspan(65, 80, alpha=0.2, color='yellow', label='Stress')
            ax.axhspan(50, 65, alpha=0.2, color='orange', label='Anxiety')
            ax.axhspan(20, 50, alpha=0.2, color='red', label='Depression')
            ax.axhspan(0, 20, alpha=0.2, color='darkred', label='Suicidal')
            
            ax.legend(loc='lower left')
            st.pyplot(fig)
        
        st.markdown("### Analysis History")
        display_df = history_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['score'] = display_df['score'].round(1)
        display_df.columns = ['Date & Time', 'Score', 'Category']
        
        st.dataframe(display_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.current_score = None
            st.success("History cleared successfully!")
            st.experimental_rerun()

elif page == "About":
    st.markdown('<p class="main-header">About AI Mental Health Monitor</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    
    The AI Mental Health Monitor is an application designed to detect early signs of 
    depression, anxiety, and stress using a custom-trained Logistic Regression machine learning model.
    
    ### How It Works
    
    #### 1. Machine Learning Classification
    The core of this application is a Machine Learning model (`best_mental_health_model.pkl`) trained on clinical/text data to classify inputs into categories like Anxiety, Depression, Stress, Suicidal, or Normal. It uses a TF-IDF vectorizer to process text.
    
    #### 2. NLP Analysis
    - **VADER Sentiment Analysis**: Evaluates the overall emotional tone (positive/negative/neutral)
    - **Linguistic Markers**: Detects rule-based patterns associated with mental health concerns
    - **Text Metrics**: Analyzes writing complexity
    
    ### Privacy & Security
    
    Your privacy is our top priority. All analysis is performed within the isolated server container and no text is persistently logged or sold to third parties.
    
    ### Disclaimer
    
    This application is an educational AI demonstration and is **not a substitute for professional mental health care**. If you're experiencing mental health concerns, please consult with a qualified healthcare provider.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>" +
    "AI Mental Health Monitor | Powered by Scikit-Learn | " +
    f"© {datetime.datetime.now().year}</p>",
    unsafe_allow_html=True
)