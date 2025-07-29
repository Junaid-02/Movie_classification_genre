#!/usr/bin/env python3
"""
Movie Genre Classification Model Training Script
This script trains a machine learning model to classify movies by genre based on plot descriptions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def load_data():
    """Load the movie dataset"""
    try:
        # Try to load from data directory first
        df = pd.read_csv('data/wiki_movie_plots_deduped.csv')
    except FileNotFoundError:
        # Fallback to root directory
        df = pd.read_csv('wiki_movie_plots_deduped.csv')
    
    return df

def preprocess_data(df):
    """Preprocess the movie data"""
    # Remove rows with missing plot or genre
    df = df.dropna(subset=['Plot', 'Genre'])
    
    # Filter for common genres (at least 100 movies)
    genre_counts = df['Genre'].value_counts()
    common_genres = genre_counts[genre_counts >= 100].index
    df = df[df['Genre'].isin(common_genres) & (df['Genre'] != 'unknown')]
    
    # Genre mapping for consistency
    genre_mapping = {
        'science fiction': 'sci-fi',
        'sci-fi': 'sci-fi',
        'comedy drama': 'comedy-drama',
        'comedy, drama': 'comedy-drama',
        'comedy-drama': 'comedy-drama',
        'romantic comedy': 'romance',
        'romantic drama': 'romance',
        'animated': 'animation'
    }
    df['Genre'] = df['Genre'].replace(genre_mapping)
    
    return df

def train_model(X_train, y_train):
    """Train the classification model"""
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    
    return vectorizer, classifier

def save_model(vectorizer, classifier, model_path='models/movie_classifier.pkl'):
    """Save the trained model"""
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'vectorizer': vectorizer,
        'classifier': classifier
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path}")

def main():
    """Main training function"""
    print("Loading movie dataset...")
    df = load_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Genres: {df['Genre'].value_counts().head()}")
    
    # Split data
    X = df['Plot']
    y = df['Genre']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training model...")
    vectorizer, classifier = train_model(X_train, y_train)
    
    # Evaluate model
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    save_model(vectorizer, classifier)
    
    # Save test data for later evaluation
    test_data = pd.DataFrame({
        'plot': X_test,
        'genre': y_test,
        'predicted_genre': y_pred
    })
    test_data.to_csv('data/test_data.txt', index=False)
    print("Test data saved to data/test_data.txt")

if __name__ == "__main__":
    main() 