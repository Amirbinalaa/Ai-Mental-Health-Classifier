import os
import joblib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from app.backend.text_analysis.analyzer import TextAnalyzer
from app.backend.models.scoring import MentalHealthScorer
from app.backend.models.suggestions import SuggestionEngine

app = Flask(__name__)

# Load ML models on startup
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('best_mental_health_model.pkl')
    print("ML models loaded successfully.")
    ml_models_loaded = True
except Exception as e:
    print(f"Error loading ML models: {e}")
    ml_models_loaded = False

# Initialize analysis engines
text_analyzer = TextAnalyzer()
scorer = MentalHealthScorer()
suggestion_engine = SuggestionEngine()

def adjust_ml_probabilities(text, ml_prediction, analysis_results):
    """Refine model probabilities, ensuring safety overrides are met but trusting the balanced model prediction"""
    if not ml_prediction or "probabilities" not in ml_prediction:
        return ml_prediction
        
    probs = ml_prediction["probabilities"].copy()
    sentiment = analysis_results.get("sentiment", {})
    compound = sentiment.get("compound", 0.0)
    markers = analysis_results.get("markers", {})
    
    dep_markers = markers.get("depression", 0)
    anx_markers = markers.get("anxiety", 0)
    str_markers = markers.get("stress", 0)
    risk_markers = markers.get("severe_risk", 0)
    
    # 1. Base Class Identification (trust the balanced ML model as the baseline)
    final_class = ml_prediction["class"]
    
    # Check for direct emergency triggers (Suicidal)
    if risk_markers > 0 or probs.get("Suicidal", 0.0) > 0.15:
        final_class = "Suicidal"
    # Check if highly positive with zero negative markers -> Normal
    elif compound >= 0.25 and dep_markers == 0 and anx_markers == 0 and str_markers == 0 and risk_markers == 0:
        final_class = "Normal"
            
    # 2. Probability Redistribution
    # Re-normalize probabilities so that the chosen final_class has dominant probability
    if final_class in probs:
        target_prob = 0.80 if final_class != "Normal" else 0.90
        target_prob = max(target_prob, probs[final_class])
        
        probs[final_class] = target_prob
        remaining = 1.0 - target_prob
        total_other = sum(v for k, v in probs.items() if k != final_class)
        
        if total_other > 0:
            for k in probs:
                if k != final_class:
                    probs[k] = (probs[k] / total_other) * remaining
        else:
            for k in probs:
                if k != final_class:
                    probs[k] = remaining / (len(probs) - 1)
                    
    # Force sum to exactly 1.0
    total_sum = sum(probs.values())
    if total_sum > 0:
        for k in probs:
            probs[k] /= total_sum
            
    ml_prediction["class"] = final_class
    ml_prediction["probabilities"] = probs
    return ml_prediction

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    # 1. Base NLTK text analysis
    analysis_results = text_analyzer.analyze(text)
    
    # 2. Scikit-learn ML Prediction
    ml_prediction = None
    if ml_models_loaded:
        try:
            X = vectorizer.transform([text])
            pred_class = str(model.predict(X)[0])
            pred_proba = [float(p) for p in model.predict_proba(X)[0].tolist()]
            classes = [str(c) for c in model.classes_.tolist()]
            
            ml_prediction = {
                'class': pred_class,
                'probabilities': dict(zip(classes, pred_proba))
            }
            
            # Apply hybrid NLP corrections to resolve ML model deficiencies
            ml_prediction = adjust_ml_probabilities(text, ml_prediction, analysis_results)
            analysis_results['ml_prediction'] = ml_prediction
        except Exception as e:
            print(f"ML Prediction error: {e}")

    # 3. Calculate Score (incorporating ML predictions if loaded)
    score_result = scorer.score_text_analysis(analysis_results, ml_prediction)
        
    # 4. Get Suggestions
    suggestions = suggestion_engine.get_suggestions(score_result, analysis_results)
    
    response = {
        'score': score_result['score'],
        'category': score_result['category'],
        'analysis': analysis_results,
        'suggestions': suggestions
    }
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # When deployed to Railway, gunicorn will use app:app instead of this block
    print("Starting AI Mental Health Monitor Backend...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
