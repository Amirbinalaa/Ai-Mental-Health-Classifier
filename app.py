import streamlit as st
import joblib
import os
import re
import datetime
import nltk
import numpy as np

# Download essential tokenization assets automatically on runtime
nltk.download('punkt')
nltk.download('punkt_tab')

# Verify and import the custom Rule-Based TextAnalyzer engine
try:
    from analyzer import TextAnalyzer
except ImportError:
    st.error("🚨 Critical Error: `analyzer.py` is missing from the root directory! Please ensure it sits alongside this app.py file.")

# 1. Streamlit Application Core Configuration
st.set_page_config(
    page_title="AI Mental Health Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Optimized Preprocessing Pipeline (Sanitizes input while keeping critical token forms)
def preprocess_text(text):
    text = str(text).lower()
    
    # Replace standard and curly apostrophes with space to isolate contractions safely (e.g., I'm -> i m)
    text = text.replace("'", " ").replace("’", " ")
    
    # Strip web links and explicit URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Drop special characters, preserving only pure characters and spacing structure
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return ' '.join(text.split())

# 3. Cached Resource Loader for Fast Component Fetching
@st.cache_resource
def load_components():
    model_path = "best_mental_health_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        try:
            analyzer = TextAnalyzer()
        except:
            analyzer = None
        return model, vectorizer, analyzer
    else:
        return None, None, None

model, vectorizer, analyzer = load_components()

# --- User Interface Setup ---

# Sidebar Context Navigation Panel
with st.sidebar:
    st.markdown("### 🧠 About the System")
    st.info(
        "This system implements a Hybrid NLP Architecture. It leverages a highly balanced "
        "Machine Learning model (Logistic Regression Engine) alongside rule-based Linguistic "
        "Analysis to cross-verify psychological indicators from text inputs."
    )
    st.markdown("---")
    st.markdown("### 🔒 Privacy & Isolation")
    st.caption(
        "All queries are processed in memory within your isolated application environment. "
        "No data is permanently stored, tracked, or saved."
    )

# Page Branding Header
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>🧠 AI Mental Health Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Intelligent Text-Based Screening & Hybrid NLP Analytics Platform</p>", unsafe_allow_html=True)
st.markdown("---")

# Verify asset dependencies state before running interactive segments
if model is None or vectorizer is None:
    st.error("🚨 Deployment Error: Model artifact files (`best_mental_health_model.pkl` or `tfidf_vectorizer.pkl`) could not be found in the root space directory!")
else:
    # Segment UI Layout splitting into user fields and contextual guides
    col_input, col_info = st.columns([1.2, 0.8])
    
    with col_input:
        st.markdown("### 📝 Input Segment")
        user_input = st.text_area(
            "Describe your current emotional status or paste text indicators below:", 
            height=200, 
            placeholder="Type your thoughts here... e.g., I have been feeling completely drained and exhausted lately, finding it hard to sleep..."
        )
        
        btn_analyze = st.button("Execute Full Screening", type="primary", use_container_width=True)
        
    with col_info:
        st.markdown("### 📖 Analytics Guidelines")
        st.markdown(
            "1. **Predicted Status:** The primary classification computed directly via the ML Model.\n"
            "2. **Linguistic Markers:** Rule-based keyword frequency counts extracted from your text using regex criteria.\n"
            "3. **Structural Characteristics:** Metrics showing text complexity, diversity, and internal focus parameters."
        )

    # Core Assessment Pipeline Trigger Block
    if btn_analyze:
        if not user_input.strip():
            st.warning("⚠️ Input required. Please enter or paste some text before starting the screening.")
        else:
            with st.spinner("Processing structural parsing and model pipeline predictions..."):
                
                # A. Run rule-based analyzer first to provide metrics for the decision engine
                if analyzer:
                    analysis = analyzer.analyze(user_input)
                else:
                    analysis = None
                
                # B. Preprocess and compute statistical ML model inferences
                cleaned_text = preprocess_text(user_input)
                text_vectorized = vectorizer.transform([cleaned_text])
                raw_prediction = str(model.predict(text_vectorized)[0]).strip().lower()
                
                # C. Hybrid Decision Fusion Layer (Cross-checks statistical classifications with rules)
                if analysis and "markers" in analysis:
                    markers = analysis["markers"]
                    
                    # Priority 1: Instant Override if distinct self-harm flags are isolated
                    if markers.get("severe_risk", 0) > 0:
                        predicted_status = "Suicidal"
                    
                    # Priority 2: Direct routing if stress signals dominate or match depression flags
                    elif markers.get("stress", 0) >= 2 and markers.get("stress", 0) >= markers.get("depression", 0):
                        predicted_status = "Stress"
                    
                    # Priority 3: Routing to Anxiety if anxiety counters take local statistical dominance
                    elif markers.get("anxiety", 0) >= 2 and markers.get("anxiety", 0) >= markers.get("depression", 0):
                        predicted_status = "Anxiety"
                    
                    # Priority 4: Complete system discharge if no toxic linguistic cues exist anywhere
                    elif markers.get("depression", 0) == 0 and markers.get("anxiety", 0) == 0 and markers.get("stress", 0) == 0:
                        predicted_status = "Normal"
                    
                    # Fallback default boundary classification matching the global training environment
                    else:
                        predicted_status = raw_prediction.capitalize()
                else:
                    # Generic fallback if custom rule analyzer instance fails internally
                    predicted_status = raw_prediction.capitalize()
                
                # Format final evaluation status name safely
                predicted_status = predicted_status.capitalize()
                
                # Render Assessment Segment Visuals
                st.markdown("## 📊 Comprehensive Assessment Report")
                st.success(f"### 🎯 Model Classification Outcome: **{predicted_status}**")
                
                # UI Layout Expansion for Heuristic Metrics Breakdown
                if analysis and "markers" in analysis:
                    st.write("---")
                    st.markdown("### 🔍 Extracted Linguistic Markers Count")
                    
                    # Distribute counts across an isolated 4-column sub-layout Grid
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Depression Signals", analysis['markers'].get('depression', 0))
                    m_col2.metric("Anxiety Signals", analysis['markers'].get('anxiety', 0))
                    m_col3.metric("Stress Signals", analysis['markers'].get('stress', 0))
                    
                    # Trigger alert styling parameters if high-risk flags are caught
                    severe_count = analysis['markers'].get('severe_risk', 0)
                    if severe_count > 0:
                        m_col4.metric("🚨 High Severity Signals", severe_count, delta="Requires Attention", delta_color="inverse")
                    else:
                        m_col4.metric("High Severity Signals", severe_count)
                        
                    # Build reporting profiles for general structural statistics
                    if "metrics" in analysis:
                        st.markdown("### 📈 Text Framework Properties")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        stat_col1.write(f"**Total Clean Word Count:** {analysis['metrics'].get('word_count', 0)}")
                        stat_col2.write(f"**Linguistic Diversity Index:** {analysis['metrics'].get('linguistic_diversity', 0)}")
                        stat_col3.write(f"**First-Person Pronouns (Self-Focus):** {analysis['metrics'].get('first_person_ratio', 0)}")
                    
                    # Present explicit high-weight keywords captured inside text strings
                    if "key_phrases" in analysis and analysis["key_phrases"]:
                        st.markdown("**Dominant Key Phrases Extracted:**")
                        st.caption(", ".join(analysis["key_phrases"]))
                        
                else:
                    st.warning("⚠️ Model prediction served successfully. Rule-based NLP metrics could not be rendered because analyzer.py did not respond.")

# System Footer Architecture
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: gray;'>AI Mental Health Monitor | Hybrid Architecture (ML + NLP Rules) | © {datetime.datetime.now().year}</p>",
    unsafe_allow_html=True
)
