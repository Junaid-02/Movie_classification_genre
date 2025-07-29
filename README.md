# Movie Genre Classification Project

A machine learning project that classifies movies by genre based on their plot descriptions using Natural Language Processing (NLP) and TF-IDF vectorization.

## 🎬 Project Overview

This project implements a movie genre classifier that can predict the genre of a movie based on its plot summary. The system uses:
- **TF-IDF Vectorization** for text feature extraction
- **Multinomial Naive Bayes** classifier for genre prediction
- **Flask Web Application** for easy interaction
- **Wikipedia Movie Plots Dataset** with 21,380 movies

## 📊 Model Performance

- **Accuracy:** 44.93%
- **Dataset Size:** 21,380 movies
- **Genres Supported:** 18 different genres including drama, comedy, horror, action, sci-fi, and more
- **Training Time:** ~30 seconds

## 🏗️ Project Structure

```
Movie_classification_genre/
├── data/                          # Data files
│   ├── wiki_movie_plots_deduped.csv    # Main dataset
│   ├── train_data.txt                  # Training data (generated)
│   └── test_data.txt                   # Test data (generated)
├── models/                        # Trained models
│   └── movie_classifier.pkl       # Trained classifier model
├── notebooks/                     # Jupyter notebooks
│   ├── Movie_Classification_genre.ipynb  # Original analysis
│   └── Movie_reviews.ipynb             # Sentiment analysis
├── templates/                     # Web app templates
│   └── index.html                # Web interface
├── app.py                        # Flask web application
├── train_model.py                # Model training script
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Movie_classification_genre
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already trained)
   ```bash
   python train_model.py
   ```

4. **Start the web application**
   ```bash
   python app.py
   ```

5. **Open your browser and visit**
   ```
   http://localhost:5000
   ```

## 📖 Usage

### Web Application

1. **Open the web interface** at `http://localhost:5000`
2. **Enter a movie plot** in the text area
3. **Click "Predict Genre"** to get the prediction
4. **View results** including:
   - Predicted genre
   - Confidence score
   - Top 5 genre probabilities

### Example Usage

**Input Plot:**
```
A young wizard discovers he has magical powers and must save the world from an evil wizard who threatens to destroy everything.
```

**Expected Output:**
- **Predicted Genre:** fantasy
- **Confidence:** 85.2%
- **Top Genres:** fantasy, adventure, drama, sci-fi, action

### Programmatic Usage

```python
import pickle

# Load the trained model
with open('models/movie_classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
vectorizer = model_data['vectorizer']
classifier = model_data['classifier']

# Make a prediction
plot = "A detective solves a mysterious crime in a small town."
plot_tfidf = vectorizer.transform([plot])
prediction = classifier.predict(plot_tfidf)[0]
confidence = max(classifier.predict_proba(plot_tfidf)[0])

print(f"Predicted Genre: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

## 🔧 Technical Details

### Model Architecture

- **Text Preprocessing:** TF-IDF vectorization with 5000 features
- **Classifier:** Multinomial Naive Bayes
- **Feature Engineering:** Stop words removal, text normalization
- **Data Split:** 80% training, 20% testing

### Supported Genres

The model can classify movies into these genres:
- Action, Adventure, Animation, Anime
- Biography, Comedy, Comedy-drama, Crime
- Crime drama, Drama, Family, Fantasy
- Film noir, Horror, Sci-fi, Suspense
- Thriller, War, Western

### Dataset Information

- **Source:** Wikipedia Movie Plots Dataset
- **Size:** 21,380 movies
- **Features:** Plot descriptions, genres, release years
- **Preprocessing:** Genre mapping and filtering for common genres

## 🛠️ Development

### Training the Model

```bash
python train_model.py
```

This script will:
1. Load and preprocess the dataset
2. Split data into training and test sets
3. Train the TF-IDF vectorizer and classifier
4. Evaluate model performance
5. Save the trained model to `models/movie_classifier.pkl`
6. Generate test data for evaluation

### Customizing the Model

You can modify `train_model.py` to:
- Change the classifier (e.g., SVM, Random Forest)
- Adjust TF-IDF parameters
- Add more preprocessing steps
- Implement cross-validation

### Web Application Development

The Flask app (`app.py`) provides:
- RESTful API endpoints
- Interactive web interface
- Real-time predictions
- Error handling

## 📈 Performance Analysis

### Current Model Performance

- **Overall Accuracy:** 44.93%
- **Best Performing Genres:** Animation (54% F1-score), Western (65% F1-score)
- **Challenges:** Some genres have low sample counts, affecting performance

### Areas for Improvement

1. **Data Augmentation:** Increase samples for underrepresented genres
2. **Feature Engineering:** Add more text features (n-grams, word embeddings)
3. **Model Selection:** Try different classifiers (SVM, Neural Networks)
4. **Ensemble Methods:** Combine multiple models for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Wikipedia Movie Plots Dataset
- Scikit-learn for machine learning tools
- Flask for web framework
- NLTK for natural language processing

## 📞 Support

If you encounter any issues or have questions:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information

---

**Happy Movie Genre Classification! 🎬✨**
