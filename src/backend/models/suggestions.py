import random

class SuggestionEngine:
    """Class for generating personalized mental health suggestions based on analysis results"""
    
    def __init__(self):
        # Define suggestion categories and their respective suggestions
        self.suggestions = {
            "relaxation": [
                "Try deep breathing exercises for 5 minutes: inhale for 4 counts, hold for 4, exhale for 6.",
                "Practice progressive muscle relaxation by tensing and releasing each muscle group.",
                "Take a 10-minute mindfulness break to focus on your present surroundings.",
                "Listen to calming music or nature sounds for a few minutes.",
                "Try the 5-4-3-2-1 grounding technique: acknowledge 5 things you see, 4 things you feel, 3 things you hear, 2 things you smell, and 1 thing you taste.",
                "Take a short walk outside and focus on the sensations around you.",
                "Practice gentle stretching or yoga poses to release physical tension.",
                "Try a guided meditation using a free app or online resource."
            ],
            
            "journaling": [
                "Write about three positive moments from your day, no matter how small.",
                "Express your current feelings in writing without judgment or censoring yourself.",
                "List three things you're grateful for today and why they matter to you.",
                "Write a letter to your future self about how you hope to feel tomorrow.",
                "Describe a challenge you're facing and brainstorm possible solutions.",
                "Reflect on a recent accomplishment and how it made you feel.",
                "Write about someone who has positively impacted your life and why.",
                "Create a list of personal strengths and how you've used them recently."
            ],
            
            "social": [
                "Reach out to a trusted friend or family member for a brief chat.",
                "Share something that's on your mind with someone you trust.",
                "Plan a small, low-pressure social activity with someone supportive.",
                "Join an online community related to one of your interests.",
                "Practice active listening in your next conversation by focusing fully on the other person.",
                "Express gratitude to someone who has helped you recently.",
                "Volunteer your time or skills to help others, even in a small way.",
                "Attend a local community event or group activity aligned with your interests."
            ],
            
            "physical": [
                "Take a 15-minute walk, focusing on your breathing and surroundings.",
                "Do a quick 5-minute stretching routine to release tension.",
                "Try a short home workout that matches your energy level today.",
                "Practice good sleep hygiene by establishing a calming bedtime routine.",
                "Stay hydrated by drinking water regularly throughout the day.",
                "Take short breaks from screens every hour to rest your eyes and mind.",
                "Try a new physical activity that you might enjoy.",
                "Focus on eating nutritious foods that support your mental wellbeing."
            ],
            
            "professional": [
                "Consider speaking with a mental health professional about how you're feeling.",
                "Explore therapy options in your area or through telehealth services.",
                "Look into mental health resources provided by your employer or school.",
                "Research support groups related to specific challenges you're facing.",
                "Consider using a mental health app with professional guidance features.",
                "Check if your health insurance covers mental health services.",
                "Explore crisis text lines or hotlines if you need immediate support.",
                "Ask your primary care provider for mental health referrals."
            ],
            
            "cognitive": [
                "Challenge negative thoughts by asking: Is this thought helpful? Is there evidence for it?",
                "Practice reframing negative situations by looking for alternative perspectives.",
                "Set small, achievable goals to build a sense of accomplishment.",
                "Create a worry time: schedule 15 minutes to address worries, then try to let them go outside that time.",
                "Identify and name your emotions specifically to help process them.",
                "Practice self-compassion by speaking to yourself as you would to a good friend.",
                "Break down overwhelming tasks into smaller, manageable steps.",
                "Create a list of healthy coping strategies you can use when feeling distressed."
            ]
        }
        
        # Define category weights based on score categories
        self.category_weights = {
            "excellent": {
                "relaxation": 0.2,
                "journaling": 0.2,
                "social": 0.3,
                "physical": 0.2,
                "professional": 0.0,
                "cognitive": 0.1
            },
            "good": {
                "relaxation": 0.2,
                "journaling": 0.2,
                "social": 0.2,
                "physical": 0.2,
                "professional": 0.0,
                "cognitive": 0.2
            },
            "moderate": {
                "relaxation": 0.3,
                "journaling": 0.2,
                "social": 0.1,
                "physical": 0.2,
                "professional": 0.1,
                "cognitive": 0.1
            },
            "concerning": {
                "relaxation": 0.3,
                "journaling": 0.2,
                "social": 0.1,
                "physical": 0.1,
                "professional": 0.2,
                "cognitive": 0.1
            },
            "poor": {
                "relaxation": 0.2,
                "journaling": 0.1,
                "social": 0.1,
                "physical": 0.1,
                "professional": 0.3,
                "cognitive": 0.2
            }
        }
    
    def get_suggestions(self, score_result, analysis_results=None, num_suggestions=3):
        """Generate personalized suggestions based on score and analysis results"""
        if not score_result or "category" not in score_result:
            category = "moderate"
        else:
            category = score_result["category"]
        
        # Check for suicidal or severe risk flags first
        is_severe_risk = False
        if analysis_results:
            severe_risk_count = analysis_results.get("markers", {}).get("severe_risk", 0)
            prob_suicidal = 0.0
            if "ml_prediction" in analysis_results and analysis_results["ml_prediction"]:
                prob_suicidal = analysis_results["ml_prediction"].get("probabilities", {}).get("Suicidal", 0.0)
            
            if severe_risk_count > 0 or prob_suicidal > 0.15:
                is_severe_risk = True
        
        # Get weights for the score category
        weights = self.category_weights.get(category, self.category_weights["moderate"])
        
        # Adjust weights based on analysis results if available
        if analysis_results:
            weights = self._adjust_weights_from_analysis(weights, analysis_results)
        
        # Select suggestion categories based on weights
        selected_categories = self._weighted_category_selection(weights, num_suggestions)
        
        # Get suggestions from each selected category
        suggestions = []
        for cat in selected_categories:
            # Get a random suggestion from the category that isn't already selected
            available_suggestions = [s for s in self.suggestions[cat] if s not in suggestions]
            if available_suggestions:
                suggestions.append(random.choice(available_suggestions))
        
        # If we don't have enough suggestions, fill with relaxation techniques
        while len(suggestions) < num_suggestions:
            relaxation_suggestions = [s for s in self.suggestions["relaxation"] if s not in suggestions]
            if not relaxation_suggestions:
                break  # No more unique suggestions available
            suggestions.append(random.choice(relaxation_suggestions))
        
        # Format the suggestions with categories
        formatted_suggestions = []
        for i, suggestion in enumerate(suggestions):
            for cat, suggs in self.suggestions.items():
                if suggestion in suggs:
                    formatted_suggestions.append({
                        "id": i + 1,
                        "category": cat,
                        "text": suggestion
                    })
                    break
        
        # Override first suggestion if severe risk is flagged
        if is_severe_risk:
            emergency_suggestion = {
                "id": 1,
                "category": "professional",
                "text": "IMMEDIATE ASSISTANCE: If you are in distress or considering self-harm, call or text 988 to connect with the Suicide & Crisis Lifeline immediately. Help is free, confidential, and available 24/7."
            }
            # Remove the first suggestion and insert the emergency one at the beginning
            if formatted_suggestions:
                formatted_suggestions[0] = emergency_suggestion
            else:
                formatted_suggestions.append(emergency_suggestion)
                
            # Ensure subsequent suggestion IDs match order
            for idx, item in enumerate(formatted_suggestions):
                item["id"] = idx + 1
        
        return formatted_suggestions
    
    def _adjust_weights_from_analysis(self, weights, analysis_results):
        """Adjust suggestion weights based on specific analysis results and ML predictions"""
        adjusted_weights = weights.copy()
        
        # 1. Parse ML predictions if available
        if "ml_prediction" in analysis_results and analysis_results["ml_prediction"]:
            ml_pred = analysis_results["ml_prediction"]
            probs = ml_pred.get("probabilities", {})
            prob_depression = probs.get("Depression", 0.0)
            prob_anxiety = probs.get("Anxiety", 0.0)
            prob_stress = probs.get("Stress", 0.0)
            prob_suicidal = probs.get("Suicidal", 0.0)
            severe_risk_count = analysis_results.get("markers", {}).get("severe_risk", 0)
            
            # If high depression probability
            if prob_depression > 0.3:
                adjusted_weights["journaling"] = min(1.0, adjusted_weights["journaling"] + prob_depression * 0.3)
                adjusted_weights["professional"] = min(1.0, adjusted_weights["professional"] + prob_depression * 0.2)
                
            # If high anxiety probability
            if prob_anxiety > 0.3:
                adjusted_weights["relaxation"] = min(1.0, adjusted_weights["relaxation"] + prob_anxiety * 0.3)
                adjusted_weights["cognitive"] = min(1.0, adjusted_weights["cognitive"] + prob_anxiety * 0.2)
                
            # If high stress probability
            if prob_stress > 0.3:
                adjusted_weights["physical"] = min(1.0, adjusted_weights["physical"] + prob_stress * 0.2)
                adjusted_weights["relaxation"] = min(1.0, adjusted_weights["relaxation"] + prob_stress * 0.2)
                
            # If suicidal ideation or severe risk is registered, prioritize professional suggestions
            if prob_suicidal > 0.15 or severe_risk_count > 0:
                adjusted_weights["professional"] = 1.0
                adjusted_weights["relaxation"] = 0.0
                adjusted_weights["journaling"] = 0.0
                adjusted_weights["social"] = 0.0
                adjusted_weights["physical"] = 0.0
                adjusted_weights["cognitive"] = 0.0
                return adjusted_weights
        
        # 2. Heuristic text marker adjustments as fallback/secondary influence
        if "sentiment" in analysis_results:
            sentiment = analysis_results["sentiment"]["compound"]
            if sentiment < -0.5:
                adjusted_weights["professional"] = min(1.0, adjusted_weights["professional"] + 0.1)
                adjusted_weights["cognitive"] = min(1.0, adjusted_weights["cognitive"] + 0.1)
            
            if "markers" in analysis_results:
                markers = analysis_results["markers"]
                if markers.get("depression", 0) > 2:
                    adjusted_weights["journaling"] = min(1.0, adjusted_weights["journaling"] + 0.1)
                    adjusted_weights["professional"] = min(1.0, adjusted_weights["professional"] + 0.1)
                
                if markers.get("anxiety", 0) > 2:
                    adjusted_weights["relaxation"] = min(1.0, adjusted_weights["relaxation"] + 0.1)
                    adjusted_weights["cognitive"] = min(1.0, adjusted_weights["cognitive"] + 0.1)
                    
        self._normalize_weights(adjusted_weights)
        return adjusted_weights
    
    def _normalize_weights(self, weights):
        """Normalize weights to ensure they sum to 1.0"""
        total = sum(weights.values())
        if total > 0:
            for category in weights:
                weights[category] /= total
    
    def _weighted_category_selection(self, weights, num_selections):
        """Select categories based on their weights without replacement"""
        categories = list(weights.keys())
        selected_categories = []
        
        # Make a copy of weights so we can modify it
        temp_weights = weights.copy()
        
        for _ in range(num_selections):
            total_weight = sum(temp_weights.values())
            if total_weight <= 0:
                # If all remaining weights are 0, pick a random unselected category
                unselected = [c for c in categories if c not in selected_categories]
                if unselected:
                    chosen = random.choice(unselected)
                    selected_categories.append(chosen)
                    temp_weights.pop(chosen, None)
                continue
                
            # Pick a category based on current weights
            r = random.uniform(0, total_weight)
            upto = 0
            for category, weight in list(temp_weights.items()):
                if upto + weight >= r:
                    selected_categories.append(category)
                    temp_weights.pop(category, None)
                    break
                upto += weight
                
        return selected_categories