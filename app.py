import streamlit as st
import joblib
import os
import re
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Verify and import your original TextAnalyzer from analyzer.py
try:
    from analyzer import TextAnalyzer
except ImportError:
    st.error("🚨 Critical Error: `analyzer.py` is missing from the root directory! Please make sure it is placed right next to this app.py file.")

# 1. Page Configuration for a Professional Layout
st.set_page_config(
    page_title="AI Mental Health Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Download required NLTK data dependencies for text preprocessing on the server
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# 3. Text Preprocessing Pipeline (Identical to your optimized Jupyter Notebook)
def preprocess_text(text):
    text = str(text).lower()
    # Remove URLs and links (essential to avoid web noise)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters except letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Return directly without touching stop words or lemmatizer
    return ' '.join(text.split())

# 4. Load Models and Backend Analytics with Caching for Optimal Performance
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

# Sidebar Navigation / Application Information
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

# App Core Header
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>🧠 AI Mental Health Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Intelligent Text-Based Screening & Hybrid NLP Analytics Platform</p>", unsafe_allow_html=True)
st.markdown("---")

# Verify model files deployment status before rendering input blocks
if model is None or vectorizer is None:
    st.error("🚨 Deployment Error: Model artifact files (`best_mental_health_model.pkl` or `tfidf_vectorizer.pkl`) could not be found in the root space directory!")
else:
    # Divide layout into interactive inputs and reading documentation guidelines
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

    # Trigger processing execution pipeline upon button engagement
    if btn_analyze:
        if not user_input.strip():
            st.warning("⚠️ Input required. Please enter or paste some text before starting the screening.")
        else:
            with st.spinner("Processing structural parsing and model pipeline predictions..."):
                
                # A. Apply preprocessing and inference through the ML model
                cleaned_text = preprocess_text(user_input)
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)[0]
                
                # B. Execute heuristic text analyzer metrics
                if analyzer:
                    analysis = analyzer.analyze(user_input)
                else:
                    analysis = None
                
                st.markdown("## 📊 Comprehensive Assessment Report")
                
                # Dynamic visual rendering for prediction outcome
                st.success(f"### 🎯 Model Classification Outcome: **{prediction}**")
                
                # Render linguistic heuristics if the backend analyzer module responded successfully
                if analysis and "markers" in analysis:
                    st.write("---")
                    st.markdown("### 🔍 Extracted Linguistic Markers Count")
                    
                    # Distribute counts across a 4-column row layout
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    m_col1.metric("Depression Signals", analysis['markers'].get('depression', 0))
                    m_col2.metric("Anxiety Signals", analysis['markers'].get('anxiety', 0))
                    m_col3.metric("Stress Signals", analysis['markers'].get('stress', 0))
                    
                    # Apply risk warning format adjustments if severe indicators are isolated
                    severe_count = analysis['markers'].get('severe_risk', 0)
                    if severe_count > 0:
                        m_col4.metric("🚨 High Severity Signals", severe_count, delta="Requires Attention", delta_color="inverse")
                    else:
                        m_col4.metric("High Severity Signals", severe_count)
                        
                    # Structure text framework attributes and diversity statistics
                    if "metrics" in analysis:
                        st.markdown("### 📈 Text Framework Properties")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        stat_col1.write(f"**Total Clean Word Count:** {analysis['metrics'].get('word_count', 0)}")
                        stat_col2.write(f"**Linguistic Diversity Index:** {analysis['metrics'].get('linguistic_diversity', 0)}")
                        stat_col3.write(f"**First-Person Pronouns (Self-Focus):** {analysis['metrics'].get('first_person_ratio', 0)}")
                    
                    # Highlight major key phrase elements extracted
                    if "key_phrases" in analysis and analysis["key_phrases"]:
                        st.markdown("**Dominant Key Phrases Extracted:**")
                        st.caption(", ".join(analysis["key_phrases"]))
                        
                else:
                    st.warning("⚠️ Model prediction served successfully. Rule-based NLP metrics could not be rendered because analyzer.py did not respond.")

# Page Layout Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: gray;'>AI Mental Health Monitor | Hybrid Architecture (ML + NLP Rules) | © {datetime.datetime.now().year}</p>",
    unsafe_allow_html=True
)
