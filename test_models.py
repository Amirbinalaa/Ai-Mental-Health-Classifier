import joblib

try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Vectorizer loaded successfully. Type:", type(vectorizer))
except Exception as e:
    print("Error loading vectorizer:", e)

try:
    model = joblib.load('best_mental_health_model.pkl')
    print("Model loaded successfully. Type:", type(model))
    print("Model classes (if applicable):", getattr(model, 'classes_', 'No classes attribute'))
    
    # Test prediction
    text = ["I am feeling very sad and anxious today."]
    X = vectorizer.transform(text)
    pred = model.predict(X)
    print("Prediction for test text:", pred)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)
        print("Prediction probability:", prob)
except Exception as e:
    print("Error loading model or predicting:", e)
