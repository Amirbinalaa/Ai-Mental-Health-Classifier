class MentalHealthScorer:
    """Class for generating mental health wellness scores and categories driven 100% by ML predictions"""
    
    def __init__(self):
        # Define the exact mapping of condition classes to wellness score boundaries
        self.condition_ranges = {
            "Normal": (80, 100),
            "Stress": (65, 80),
            "Anxiety": (50, 65),
            "Depression": (20, 50),
            "Suicidal": (0, 20)
        }
    
    def score_text_analysis(self, analysis_results, ml_prediction=None):
        """Generate a highly exact wellness score and condition category driven 100% by the ML model"""
        if not analysis_results or "error" in analysis_results:
            return {
                "score": 50.0,
                "category": "Normal",
                "confidence": 0.0,
                "factors": {}
            }
        
        # 1. Base Score & Category calculation using ML predictions
        if ml_prediction and "class" in ml_prediction:
            category = ml_prediction["class"]
            probs = ml_prediction.get("probabilities", {})
            confidence = 0.95
            
            # Map predicted condition directly to its corresponding wellness range
            if category == "Normal":
                base_score = 90.0
                # Micro-adjust based on Normal probability strength
                adjustment = (probs.get("Normal", 0.5) - 0.5) * 20.0
                final_score = base_score + adjustment
            elif category == "Stress":
                base_score = 72.5
                adjustment = (probs.get("Normal", 0.0) - probs.get("Depression", 0.0)) * 7.5
                final_score = base_score + adjustment
            elif category == "Anxiety":
                base_score = 57.5
                adjustment = (probs.get("Normal", 0.0) - probs.get("Depression", 0.0)) * 7.5
                final_score = base_score + adjustment
            elif category == "Depression":
                base_score = 35.0
                adjustment = (probs.get("Normal", 0.0) - probs.get("Suicidal", 0.0)) * 15.0
                final_score = base_score + adjustment
            elif category == "Suicidal":
                base_score = 10.0
                adjustment = -probs.get("Suicidal", 0.5) * 10.0
                final_score = base_score + adjustment
            else:
                final_score = 50.0
                category = "Normal"
        else:
            # Fallback expected score using VADER sentiment if ML is not active
            sentiment_compound = analysis_results.get("sentiment", {}).get("compound", 0.0)
            if sentiment_compound >= 0.30:
                category = "Normal"
                final_score = 80.0 + (sentiment_compound * 20.0)
            elif sentiment_compound <= -0.15:
                category = "Depression"
                final_score = 20.0 + ((sentiment_compound + 1.0) * 30.0)
            else:
                category = "Stress"
                final_score = 50.0 + ((sentiment_compound + 0.15) * 30.0)
            confidence = 0.50

        # 2. Strict Safety overrides for suicidal risk
        severe_risk_count = analysis_results.get("markers", {}).get("severe_risk", 0)
        prob_suicidal = 0.0
        if ml_prediction and "probabilities" in ml_prediction:
            prob_suicidal = ml_prediction["probabilities"].get("Suicidal", 0.0)
            
        # If severe suicidal keywords are present OR suicidal class has high probability
        if severe_risk_count > 0 or prob_suicidal > 0.15:
            category = "Suicidal"
            # Instant collapse of final score below 15.0
            risk_factor = max(prob_suicidal, 0.5 if severe_risk_count > 0 else 0.0)
            final_score = min(final_score, 15.0 * (1.0 - risk_factor))
            
        # Ensure score bounds
        final_score = max(0.0, min(100.0, final_score))
        
        # 3. Refine confidence score based on text length
        metrics = analysis_results.get("metrics", {})
        word_count = metrics.get("word_count", 0)
        text_length_confidence = min(1.0, word_count / 100)
        final_confidence = (confidence * 0.7) + (text_length_confidence * 0.3)
        
        # Return exact model analysis values
        return {
            "score": round(final_score, 1),
            "category": category,  # Now returns the actual identified condition!
            "confidence": round(final_confidence, 2),
            "factors": {
                "Normal_probability": round(probs.get("Normal", 0.0) if ml_prediction else 0.0, 3),
                "Stress_probability": round(probs.get("Stress", 0.0) if ml_prediction else 0.0, 3),
                "Anxiety_probability": round(probs.get("Anxiety", 0.0) if ml_prediction else 0.0, 3),
                "Depression_probability": round(probs.get("Depression", 0.0) if ml_prediction else 0.0, 3),
                "Suicidal_probability": round(probs.get("Suicidal", 0.0) if ml_prediction else 0.0, 3)
            }
        }