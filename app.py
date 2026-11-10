from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tempfile
import uuid
from user_manager import user_manager
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize user management system and create default admin if needed
user_manager.create_default_admin()

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_bert_sentiment_model():
    """Load the fine-tuned RoBERTa model and tokenizer for sentiment analysis with sarcasm awareness"""
    try:
        # First try to load the new roberta_sarcasm_aware3 model
        model_dir = 'models/roberta_sarcasm_aware3'
        label_map_path = 'models/roberta_sarcasm_aware3/label_map.pkl'

        if os.path.exists(model_dir) and os.path.exists(label_map_path):
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            with open(label_map_path, 'rb') as f:
                label_map = pickle.load(f)
            inv_label_map = {v: k for k, v in label_map.items()}
            
            print("RoBERTa sarcasm-aware sentiment model loaded from models/roberta_sarcasm_aware3.")
            return model, tokenizer, inv_label_map
        
        # Fallback to the old model if new one doesn't exist
        model_dir = 'roberta-sarcasm-aware2/roberta_sentiment_model'
        label_map_path = 'roberta-sarcasm-aware2/label_map.pkl'

        if os.path.exists(model_dir) and os.path.exists(label_map_path):
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
            with open(label_map_path, 'rb') as f:
                label_map = pickle.load(f)
            inv_label_map = {v: k for k, v in label_map.items()}
            
            print("RoBERTa sarcasm-aware sentiment model loaded from roberta-sarcasm-aware2 (fallback).")
            return model, tokenizer, inv_label_map
        
        print("Warning: No sarcasm-aware model found. Please train the model first.")
        return None, None, None
            
    except Exception as e:
        print(f"Error loading RoBERTa sentiment model: {str(e)}")
        return None, None, None

bert_model, bert_tokenizer, bert_inv_label_map = load_bert_sentiment_model()

def is_sarcasm_aware_model():
    """Check if the loaded model is the sarcasm-aware version"""
    try:
        # Check if the model has the <SARCASTIC> token in its vocabulary
        if bert_model is not None and bert_tokenizer is not None:
            # Check if the model has the <SARCASTIC> token in its vocabulary
            if '<SARCASTIC>' in bert_tokenizer.get_vocab():
                return True
        return False
    except:
        return False

def validate_gmail(email):
    """Validate if email is a valid Gmail address"""
    # Gmail pattern: username@gmail.com
    gmail_pattern = r'^[a-zA-Z0-9._%+-]+@gmail\.com$'
    return re.match(gmail_pattern, email) is not None

def validate_password(password):
    """Validate password according to policy"""
    if len(password) > 10:
        return False, "Password must be 10 characters or less"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    return True, "Password is valid"

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

def analyze_sentiment(text):
    """Analyze sentiment using the fine-tuned RoBERTa model with sarcasm awareness"""
    if not text or text.strip() == '':
        return 'Neutral'
    if bert_model is None or bert_tokenizer is None or bert_inv_label_map is None:
        print("Warning: RoBERTa sentiment model not loaded. Please train the model first.")
        return 'Neutral'
    try:
        # Check if the model is sarcasm-aware
        is_sarcasm_aware = is_sarcasm_aware_model()
        
        # Clean text (remove HTML tags, etc.)
        cleaned_text = text
        if '<br>' in cleaned_text:
            cleaned_text = cleaned_text.replace('<br>', ' ')
        if '</br>' in cleaned_text:
            cleaned_text = cleaned_text.replace('</br>', ' ')
        if '\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n', ' ')
        if '\\' in cleaned_text:
            cleaned_text = cleaned_text.replace('\\', ' ')
        
        # Tokenize input
        inputs = bert_tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=160,
            return_tensors='pt'
        )
        
        bert_model.eval()
        with torch.no_grad():
            outputs = bert_model(**{k: v for k, v in inputs.items()})
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            label = bert_inv_label_map.get(pred, 'Neutral')
            
            # Add sarcasm detection note if using sarcasm-aware model
            if is_sarcasm_aware and '<SARCASTIC>' in text:
                print(f"Note: Sarcastic text detected and processed by sarcasm-aware model")
            
            return label
    except Exception as e:
        print(f"Error in RoBERTa sentiment analysis: {str(e)}")
        return 'Neutral'

def create_visualizations(df, sentiment_counts):
    """Create visualizations for the results"""
    # Set style
    plt.style.use('default')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bar chart
    colors_bar = ['#2E8B57', '#DC143C', '#4682B4']
    bars = ax1.bar(sentiment_counts.keys(), sentiment_counts.values(), color=colors_bar)
    ax1.set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xlabel('Sentiment', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart
    colors_pie = ['#2E8B57', '#DC143C', '#4682B4']
    wedges, texts, autotexts = ax2.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), 
                                       autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax2.set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # 3. Word cloud
    all_text = ' '.join(df['review'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         max_words=100, colormap='viridis').generate(all_text)
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('Word Cloud of Reviews', fontsize=14, fontweight='bold')
    
    # 4. Sentiment trend (if multiple reviews)
    if len(df) > 1:
        sentiment_numeric = df['sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
        ax4.plot(range(len(sentiment_numeric)), sentiment_numeric, marker='o', linewidth=2, markersize=4)
        ax4.set_title('Sentiment Trend Across Reviews', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sentiment Score', fontsize=12)
        ax4.set_xlabel('Review Index', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for trend analysis', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Sentiment Trend', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
    img_bytes.seek(0)
    plt.close()
    
    return img_bytes

@app.route('/')
def welcome():
    """Welcome page"""
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Validate Gmail address
        if not validate_gmail(email):
            flash('Please enter a valid Gmail address.', 'error')
            return render_template('login.html')
        
        # Check if user exists and password is correct
        is_valid, user_name = user_manager.verify_user(email, password)
        if is_valid:
            session['user_email'] = email
            session['user_name'] = user_name
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        name = request.form['name']
        
        # Validate Gmail address
        if not validate_gmail(email):
            flash('Please enter a valid Gmail address.', 'error')
            return render_template('register.html')
        
        # Check if user already exists
        if user_manager.get_user(email):
            flash('An account with this email already exists.', 'error')
            return render_template('register.html')
        
        # Validate password
        is_valid, message = validate_password(password)
        if not is_valid:
            flash(message, 'error')
            return render_template('register.html')
        
        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        # Create new user
        success, message = user_manager.add_user(email, password, name)
        if success:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash(f'Registration failed: {message}', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('welcome'))

@app.route('/home')
def home():
    """Home page after login"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', user_name=session.get('user_name'))

@app.route('/about')
def about():
    """About page"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    # Get model information
    model_info = {
        'model_loaded': bert_model is not None,
        'is_sarcasm_aware': is_sarcasm_aware_model(),
        'model_type': 'RoBERTa Sarcasm-Aware v3' if is_sarcasm_aware_model() else 'RoBERTa Standard',
        'model_path': 'models/roberta_sarcasm_aware3' if os.path.exists('models/roberta_sarcasm_aware3') else 'roberta-sarcasm-aware2'
    }
    
    return render_template('about.html', model_info=model_info)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """File upload and analysis page"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.endswith('.csv'):
            try:
                # Generate a unique filename for the results
                base_filename = secure_filename(file.filename)
                unique_id = uuid.uuid4().hex
                results_filename = f"{os.path.splitext(base_filename)[0]}_{unique_id}.csv"
                results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
                
                # Process the CSV in chunks to handle large files
                chunk_size = 1000  # Process 1000 rows at a time
                header_written = False

                for chunk in pd.read_csv(file, chunksize=chunk_size):
                    # Check if 'review' column exists
                    if 'review' not in chunk.columns:
                        flash('CSV file must contain a column named "review"', 'error')
                        # Clean up the partially created file if it exists
                        if os.path.exists(results_filepath):
                            os.remove(results_filepath)
                        return redirect(request.url)
                    
                    # Preprocess and analyze
                    chunk['processed_review'] = chunk['review'].apply(preprocess_text)
                    chunk['sentiment'] = chunk['processed_review'].apply(analyze_sentiment)
                    
                    # Append processed chunk to the results file
                    chunk.to_csv(
                        results_filepath, 
                        mode='a', 
                        header=not header_written, 
                        index=False
                    )
                    header_written = True

                flash('Analysis completed successfully!', 'success')
                return redirect(url_for('results', filename=results_filename))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                # Clean up the partially created file if it exists
                if 'results_filepath' in locals() and os.path.exists(results_filepath):
                    os.remove(results_filepath)
                return redirect(request.url)
        else:
            flash('Please upload a CSV file', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/results')
def results():
    """Results page with visualizations"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    filename = request.args.get('filename')
    if not filename:
        flash('No analysis results found. Please upload a file first.', 'error')
        return redirect(url_for('upload'))

    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(results_filepath):
        flash('Analysis results not found. Please try uploading again.', 'error')
        return redirect(url_for('upload'))
        
    try:
        df = pd.read_csv(results_filepath)
        results_data = df.to_dict('records')
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        # Create visualizations
        img_bytes = create_visualizations(df, sentiment_counts)
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        
        return render_template('results.html', 
                             results=results_data,
                             sentiment_counts=sentiment_counts,
                             visualization=img_base64,
                             filename=filename)
    except Exception as e:
        flash(f'Error displaying results: {str(e)}', 'error')
        return redirect(url_for('upload'))

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    """Download results as PDF"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(results_filepath):
        flash('File not found.', 'error')
        return redirect(url_for('upload'))
    
    try:
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )
        story.append(Paragraph("ReviewRadar - Sentiment Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary
        df = pd.read_csv(results_filepath)
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        
        summary_text = f"""
        <b>Analysis Summary:</b><br/>
        Total Reviews Analyzed: {len(df)}<br/>
        Positive Reviews: {sentiment_counts.get('Positive', 0)}<br/>
        Negative Reviews: {sentiment_counts.get('Negative', 0)}<br/>
        Neutral Reviews: {sentiment_counts.get('Neutral', 0)}<br/>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Visualizations
        img_bytes = create_visualizations(df, sentiment_counts)
        img_bytes.seek(0)
        img = Image(img_bytes, width=6*inch, height=4.8*inch)
        story.append(img)
        story.append(Spacer(1, 20))
        
        # Results table
        if len(df) <= 50:  # Only show table if not too many results
            table_data = [['Review', 'Sentiment']]
            for _, row in df.iterrows():
                review_text = str(row['review'])[:100] + "..." if len(str(row['review'])) > 100 else str(row['review'])
                table_data.append([review_text, row['sentiment']])
            
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='sentiment_analysis_report.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        flash(f'Error generating PDF: {str(e)}', 'error')
        return redirect(url_for('results', filename=filename))

@app.route('/download_csv/<filename>')
def download_csv(filename):
    """Download processed CSV"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    results_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(results_filepath):
        flash('File not found.', 'error')
        return redirect(url_for('upload'))
    
    try:
        return send_file(
            results_filepath,
            as_attachment=True,
            download_name='sentiment_analysis_results.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        flash(f'Error downloading CSV: {str(e)}', 'error')
        return redirect(url_for('results', filename=filename))

@app.route('/api/model_status')
def model_status():
    """API endpoint to check model status"""
    if 'user_email' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    status = {
        'model_loaded': bert_model is not None,
        'is_sarcasm_aware': is_sarcasm_aware_model(),
        'model_type': 'RoBERTa Sarcasm-Aware v3' if is_sarcasm_aware_model() else 'RoBERTa Standard',
        'model_path': 'models/roberta_sarcasm_aware3' if os.path.exists('models/roberta_sarcasm_aware3') else 'roberta-sarcasm-aware2',
        'tokenizer_loaded': bert_tokenizer is not None,
        'label_mapping_loaded': bert_inv_label_map is not None
    }
    
    return jsonify(status)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict sentiment for individual text input"""
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            flash('Please enter some text to analyze.', 'error')
            return render_template('predict.html')
        
        try:
            # Analyze sentiment using the loaded model
            sentiment = analyze_sentiment(text)
            
            # Check if sarcasm was detected
            sarcasm_detected = '<SARCASTIC>' in text and is_sarcasm_aware_model()
            
            return render_template('predict.html', 
                                 text=text, 
                                 sentiment=sentiment,
                                 sarcasm_detected=sarcasm_detected)
        except Exception as e:
            flash(f'Error analyzing sentiment: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 