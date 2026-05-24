import streamlit as st
import joblib
import os
import re
from analyzer import TextAnalyzer # استدعاء الـ Analyzer الأصلي بتاعك

# 1. إعدادات الصفحة
st.set_page_config(page_title="AI Mental Health Monitor", page_icon="🧠", layout="centered")

# 2. تحميل المكونات مرة واحدة (Cache)
@st.cache_resource
def load_components():
    model = joblib.load("best_mental_health_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    analyzer = TextAnalyzer() 
    return model, vectorizer, analyzer

model, vectorizer, analyzer = load_components()

st.title("🧠 AI Mental Health Monitor")
st.write("تحليل الحالة النفسية بناءً على النص")

# 3. واجهة المستخدم
user_input = st.text_area("اكتب ما تشعر به هنا:", height=150, placeholder="مثلاً: أشعر بالإرهاق ولا أستطيع النوم...")

if st.button("بدء التحليل", type="primary"):
    if user_input.strip():
        with st.spinner("جاري التحليل..."):
            # A. التنبؤ بالموديل (ML)
            vec = vectorizer.transform([user_input])
            prediction = model.predict(vec)[0]
            
            # B. تحليل الـ Markers (اللي في analyzer.py)
            analysis = analyzer.analyze_text(user_input)
            
            # C. عرض النتائج
            st.success(f"### 📊 الحالة المتوقعة: {prediction}")
            
            st.write("---")
            st.write("### 🔍 تفاصيل التحليل اللغوي:")
            col1, col2, col3 = st.columns(3)
            col1.metric("مؤشر الاكتئاب", analysis['markers']['depression'])
            col2.metric("مؤشر القلق", analysis['markers']['anxiety'])
            col3.metric("مؤشر الضغط", analysis['markers']['stress'])
    else:
        st.warning("الرجاء إدخال نص للتحليل.")
