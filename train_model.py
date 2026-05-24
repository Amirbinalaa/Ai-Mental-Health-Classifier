import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import os

def main():
    print("=== AI Mental Health Monitor - Model Training ===")
    
    # 1. Load dataset
    dataset_path = 'Sentiment Analysis for Mental Health.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['statement', 'status'])
    print(f"Loaded {len(df)} valid records.")
    print("Original class distribution:")
    print(df['status'].value_counts())
    
    # 2. Solve Class Imbalance using Random Oversampling
    # Find the size of the majority class to scale the minority classes
    classes = df['status'].unique()
    max_size = df['status'].value_counts().max()
    target_size = int(max_size * 0.8) # Oversample minority classes to 80% of max class size
    
    print(f"\nOversampling minority classes to target size of {target_size} samples...")
    resampled_parts = []
    for cls in classes:
        cls_df = df[df['status'] == cls]
        if len(cls_df) < target_size:
            # Oversample with replacement
            cls_oversampled = resample(
                cls_df, 
                replace=True, 
                n_samples=target_size, 
                random_state=42
            )
            resampled_parts.append(cls_oversampled)
            print(f" - Oversampled {cls}: {len(cls_df)} -> {target_size}")
        else:
            resampled_parts.append(cls_df)
            print(f" - Kept {cls} as is: {len(cls_df)}")
            
    df_balanced = pd.concat(resampled_parts)
    print(f"Balanced dataset size: {len(df_balanced)}")
    
    # 3. Fit Vectorizer on the entire original text to have a broad vocabulary
    print("\nFitting TF-IDF Vectorizer (ngram_range=(1, 3), max_features=40000, no stop words)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), 
        max_features=40000, 
        sublinear_tf=True
    )
    # Fit on original texts so the vocabulary is realistic and not biased by duplicate texts
    vectorizer.fit(df['statement'])
    
    # 4. Transform balanced statements
    print("Transforming balanced statements to TF-IDF features...")
    X_train = vectorizer.transform(df_balanced['statement'])
    y_train = df_balanced['status']
    
    # 5. Train model
    print("Training Logistic Regression classifier (C=1.0, max_iter=1000)...")
    model = LogisticRegression(
        max_iter=1000, 
        C=1.0, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 6. Save model and vectorizer
    vectorizer_output = 'tfidf_vectorizer.pkl'
    model_output = 'best_mental_health_model.pkl'
    
    print(f"\nSaving vectorizer to {vectorizer_output}...")
    joblib.dump(vectorizer, vectorizer_output)
    
    print(f"Saving model to {model_output}...")
    joblib.dump(model, model_output)
    
    print("\nModel training and serialization completed successfully!")

if __name__ == '__main__':
    main()
