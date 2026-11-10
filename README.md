# ReviewRadar

![ReviewRadar](https://img.shields.io/badge/ReviewRadar-Sentiment%20Analysis-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Project Overview

ReviewRadar is an advanced sentiment analysis platform that helps businesses and researchers understand customer feedback through cutting-edge machine learning techniques. Built with Flask and powered by RoBERTa (a state-of-the-art transformer model), ReviewRadar processes customer reviews and provides detailed insights into sentiment patterns with sarcasm awareness.

## ✨ Features

- **AI-Powered Analysis**: Advanced RoBERTa-based sentiment classification with sarcasm detection
- **User Authentication**: Secure login/registration system with Gmail validation
- **Batch Processing**: Upload CSV files for bulk sentiment analysis
- **Real-time Prediction**: Individual text analysis for quick insights
- **Rich Visualizations**: Interactive charts, graphs, and word clouds
- **Export Options**: Download comprehensive PDF reports and CSV files
- **Modern UI**: Responsive, user-friendly interface with gradient designs
- **Secure & Private**: No permanent data storage, session-based processing

## 🏗️ Architecture

```
Review-Radar/
├── app.py                 # Main Flask application
├── user_manager.py        # User authentication system
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── base.html        # Base template with navigation
│   ├── welcome.html     # Landing page
│   ├── login.html       # Login page
│   ├── register.html    # Registration page
│   ├── home.html        # Dashboard
│   ├── upload.html      # File upload page
│   ├── predict.html     # Text prediction page
│   ├── about.html       # About page
│   └── results.html     # Results display page
├── models/              # ML models
│   └── roberta_sarcasm_aware3/  # Fine-tuned RoBERTa model
├── static/              # Static assets (CSS, JS, images)
├── uploads/             # Temporary file storage
├── screenshots/         # Application screenshots
└── docs/               # Additional documentation
```

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask 3.0.0**: Web framework
- **Flask-Login 0.6.3**: User authentication
- **Pandas 2.1.4**: Data manipulation
- **NumPy 1.26.2**: Numerical computing

### Machine Learning
- **PyTorch 2.1.1**: Deep learning framework
- **Transformers 4.35.2**: Hugging Face library
- **RoBERTa**: Pre-trained transformer model (fine-tuned for sentiment)
- **NLTK 3.8.1**: Natural language processing
- **Safetensors 0.4.1**: Safe model serialization

### Data Visualization
- **Matplotlib 3.8.2**: Plotting library
- **Seaborn 0.13.0**: Statistical visualization
- **WordCloud 1.9.2**: Word cloud generation
- **ReportLab 4.0.7**: PDF report generation

### Frontend
- **HTML5/CSS3**: Markup and styling
- **JavaScript**: Client-side interactivity
- **Font Awesome 6.0.0**: Icon library
- **Responsive Design**: Mobile-friendly interface

## 🤖 ML Models Used

### Primary Model: RoBERTa Sarcasm-Aware v3 (checkpoint-3576)
- **Base Architecture**: RoBERTa (Robustly optimized BERT approach)
- **Purpose**: Sentiment classification with sarcasm detection
- **Classes**: Positive, Negative, Neutral
- **Special Feature**: Sarcasm-aware tokenization with `<SARCASTIC>` token
- **Training Dataset**: Balanced 18k dataset (6k Negative, 6k Neutral, 5,874 Positive)
- **Performance Metrics**:
  - **Accuracy**: 80.06%
  - **F1 Score**: 80.14%
  - **F1 Macro**: 80.14%
  - **F1 Weighted**: 80.11%
  - **Training Epoch**: 2 (early stopping prevented overfitting)
- **Model Selection**: Chosen as the best-performing checkpoint from extensive training runs

### Model Specifications
- **Parameters**: 125M
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Hidden Layers**: 12
- **Max Sequence Length**: 514 tokens
- **Vocabulary Size**: 50,266 tokens
- **Model Size**: ~476 MB (model.safetensors)

### Training Configuration
- **Learning Rate**: 2e-5 with linear scheduler
- **Batch Size**: 8
- **Max Sequence Length**: 160 tokens
- **Optimizer**: AdamW with weight decay
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: Patience=2, monitoring F1 score

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Review-Radar.git
cd Review-Radar
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
The application will automatically download required NLTK data on first run, or you can manually download:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage

### Start the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Default Admin Account
- **Email**: admin@gmail.com
- **Password**: Admin123!

⚠️ **Important**: Change the default admin credentials in production!

### Features Walkthrough

#### 1. User Registration
- Navigate to the welcome page
- Click "Get Started Free"
- Enter your Gmail address (must be @gmail.com)
- Create a password (max 10 chars, uppercase, special char, number)
- Complete registration

#### 2. Login
- Enter your Gmail address and password
- Access the dashboard

#### 3. Upload and Analyze Reviews
- Click "Upload" in the navigation
- Select a CSV file with a 'review' column
- Wait for analysis to complete
- View results with visualizations
- Download PDF report or CSV file

#### 4. Individual Text Prediction
- Click "Predict" in the navigation
- Enter your review text
- Get instant sentiment prediction
- View sarcasm detection alerts

#### 5. View About Page
- Learn about the technology stack
- Understand the sentiment analysis process
- View use cases and features

## 📊 CSV File Format

Your CSV file must contain a column named `review`:

```csv
review,rating,date
"This product exceeded my expectations!",5,2024-01-15
"Disappointed with the quality",2,2024-01-16
"Good value for the price",4,2024-01-17
```

**Requirements:**
- File format: CSV
- Required column: `review`
- Optional columns: `rating`, `date`, etc.
- Maximum file size: 10MB
- Language: English (best results)

## 🔒 Security Features

- **Gmail Validation**: Only Gmail addresses accepted
- **Password Policy**: Enforced complexity requirements
- **Session Management**: Secure session handling
- **No Permanent Storage**: Files deleted after processing
- **CSRF Protection**: Form validation and sanitization
- **HTTPS Ready**: Supports secure connections

## 📸 Screenshots

Add screenshots of the application interface here:
- Welcome/Landing page
- Dashboard
- Upload interface
- Results visualization
- Prediction page

To add screenshots:
1. Place screenshot images in the `screenshots/` directory
2. Update this section with appropriate image references

## 🎯 Use Cases

- **E-commerce**: Analyze product reviews and customer feedback
- **Customer Service**: Monitor support ticket sentiment
- **Social Media**: Track brand mentions and public opinion
- **Research**: Academic and market research projects
- **Business Intelligence**: Strategic decision making
- **Market Analysis**: Competitor analysis and trends

## 🔧 Configuration

### Change Secret Key
Edit `app.py`:
```python
app.secret_key = 'your-secret-key-here'  # Change this!
```

### Change Upload Folder
Edit `app.py`:
```python
UPLOAD_FOLDER = 'uploads'  # Change if needed
```

### Modify Model Path
Edit `app.py` in `load_bert_sentiment_model()`:
```python
model_dir = 'models/roberta_sarcasm_aware3'  # Adjust path
```

## 🐛 Troubleshooting

### Model Not Loading
- Ensure model files are in the correct directory
- Check that all model files (config.json, model.safetensors, etc.) are present
- Verify sufficient disk space

### NLTK Download Issues
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Port Already in Use
Change port in `app.py`:
```python
app.run(host="0.0.0.0", port=5001)  # Use different port
```

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced sentiment metrics (emotion detection)
- [ ] Real-time dashboard with live updates
- [ ] API endpoints for integration
- [ ] Dark mode theme
- [ ] Export to more formats (Excel, JSON)
- [ ] Historical analysis and trend tracking
- [ ] Custom model training interface
- [ ] Collaborative features (team accounts)
- [ ] Advanced analytics dashboard

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

- **Project Link**: [https://github.com/yourusername/Review-Radar](https://github.com/yourusername/Review-Radar)
- **Issues**: [https://github.com/yourusername/Review-Radar/issues](https://github.com/yourusername/Review-Radar/issues)

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Facebook AI Research for RoBERTa
- Flask community for the web framework
- NLTK team for natural language processing tools

---

**Built with ❤️ using Python, Flask, and RoBERTa**
