#!/usr/bin/env python3
"""
Movie Genre Classification Web Application
A Flask web app that allows users to input movie plots and get genre predictions.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
def load_model():
    """Load the trained movie classification model"""
    try:
        with open('models/movie_classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['vectorizer'], model_data['classifier']
    except FileNotFoundError:
        print("Model file not found. Please run train_model.py first.")
        return None, None

vectorizer, classifier = load_model()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict movie genre from plot"""
    try:
        # Get plot from request
        plot = request.json.get('plot', '')
        
        if not plot:
            return jsonify({'error': 'Please provide a movie plot'}), 400
        
        # Transform the plot using the vectorizer
        plot_tfidf = vectorizer.transform([plot])
        
        # Make prediction
        predicted_genre = classifier.predict(plot_tfidf)[0]
        
        # Get prediction probabilities
        probabilities = classifier.predict_proba(plot_tfidf)[0]
        classes = classifier.classes_
        
        # Create probability dictionary
        genre_probs = dict(zip(classes, probabilities))
        
        # Sort by probability (descending)
        sorted_probs = sorted(genre_probs.items(), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'predicted_genre': predicted_genre,
            'confidence': float(max(probabilities)),
            'all_probabilities': sorted_probs[:5]  # Top 5 genres
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': classifier is not None})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create a simple HTML template
    html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Movie Genre Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        textarea { width: 100%; height: 150px; margin: 10px 0; padding: 10px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Genre Classifier</h1>
        <p>Enter a movie plot and get the predicted genre:</p>
        
        <textarea id="plot" placeholder="Enter movie plot here..."></textarea>
        <br>
        <button onclick="predictGenre()">Predict Genre</button>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        async function predictGenre() {
            const plot = document.getElementById('plot').value;
            const resultDiv = document.getElementById('result');
            
            if (!plot.trim()) {
                resultDiv.innerHTML = '<p class="error">Please enter a movie plot.</p>';
                resultDiv.style.display = 'block';
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ plot: plot })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    let html = `<h3>Prediction Results:</h3>`;
                    html += `<p><strong>Predicted Genre:</strong> <span class="success">${data.predicted_genre}</span></p>`;
                    html += `<p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>`;
                    html += `<h4>Top 5 Genres:</h4><ul>`;
                    data.all_probabilities.forEach(([genre, prob]) => {
                        html += `<li>${genre}: ${(prob * 100).toFixed(2)}%</li>`;
                    });
                    html += `</ul>`;
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                }
                
                resultDiv.style.display = 'block';
            } catch (error) {
                resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                resultDiv.style.display = 'block';
            }
        }
    </script>
</body>
</html>
'''
    
    # Write the template file
    with open('templates/index.html', 'w') as f:
        f.write(html_template)
    
    print("Starting Movie Genre Classifier Web App...")
    print("Visit http://localhost:5000 to use the application")
    app.run(debug=True, host='0.0.0.0', port=5000) 