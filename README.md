#  Fake News Detection System

A comprehensive AI-powered fake news detection system with modern web interface, real-time analysis, and interactive dashboard.

##  Features



- **AI-Powered Analysis**: Advanced machine learning model using Logistic Regression and TF-IDF vectorization
- **Real-time Detection**: Instant analysis with confidence scores and probability distributions
- **Interactive Dashboard**: Comprehensive visualizations and statistics
- **Modern UI**: Responsive design with beautiful animations and user experience
- **API Access**: RESTful endpoints for integration with other applications
- **Privacy Focused**: Secure processing without permanent data storage

##  Live Demo

The application provides:
- **Home Page**: News analysis interface with real-time results
- **Dashboard**: Analytics and visualizations of the dataset
- **About Page**: Project information and technical details

##  Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, NLTK
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript
- **UI Framework**: Custom responsive design with Font Awesome icons

##  Model Architecture

The fake news detection model uses:
- **Text Preprocessing**: NLTK for tokenization, stemming, and stop word removal
- **Feature Extraction**: TF-IDF vectorization for text representation
- **Classification**: Logistic Regression for binary classification
- **Performance**: High accuracy on balanced dataset

##  Dataset

The model is trained on a comprehensive dataset containing:
- **True News**: Verified factual news articles
- **Fake News**: Misleading or false news articles
- **Features**: Title and text content for analysis

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fakenewsdetection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

5. **Train the model** (if not already trained)
   ```bash
   python train_model.py
   ```

##  Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - The application will be available on all network interfaces

##  Usage

### Home Page
1. Navigate to the home page
2. Enter a news title and content
3. Click "Analyze News" to get instant results
4. View confidence scores and probability distributions

### Dashboard
1. Click "Dashboard" in the navigation
2. Explore dataset statistics and visualizations
3. View model performance metrics
4. Analyze data distributions

### API Usage
```bash
# Example API call
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Your news title",
    "content": "Your news content"
  }'
```

##  Features Overview

### Real-time Analysis
- Instant text processing and analysis
- Confidence scores and probability distributions
- Visual indicators for true/fake news

### Interactive Dashboard
- Dataset statistics and metrics
- Distribution charts and visualizations
- Model performance insights

### Modern UI/UX
- Responsive design for all devices
- Smooth animations and transitions
- Intuitive navigation and user flow

### Data Visualization
- Pie charts for news distribution
- Histograms for text length analysis
- Word count distributions
- Interactive charts and graphs

##  Model Performance

The Logistic Regression model achieves:
- High accuracy on balanced dataset
- Fast inference times
- Reliable confidence scoring
- Robust text preprocessing
