from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import joblib
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.metrics import MeanSquaredError
import re
import json



app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    posts = db.relationship('Post', backref='author', lazy=True)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    comments = db.relationship('Comment', backref='post', lazy=True)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)

with open('harassment_types.json', 'r') as json_file:
    harassment_types = json.load(json_file)

# Function to predict harassment type
def predict_harassment_type(text):
    text = text.lower()
    for harassment_type, keywords in harassment_types.items():
        if any(keyword in text for keyword in keywords):
            return harassment_type
    return "None"  # No harassment detected

# Define severity mapping
severity_mapping = {
    1: 'None',
    2: 'Mild',
    3: 'Severe'
}

# Load the tokenizer
tokenizer_path = "tokenizer.pkl"  # Ensure this is saved during training
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the trained models
xgboost_model = joblib.load("xgboost_model.pkl")
severity_model = load_model("severity_model.h5",custom_objects={'mse': MeanSquaredError()})

# Preprocess the input text
def preprocess_input(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#", "", text)       # Remove hashtags
    text = re.sub(r"[^\w\s]", "", text) # Remove special characters
    text = text.lower()                 # Convert to lowercase
    return text

# Routes
@app.route('/')
def home():
    """Display all posts from all users."""
    if 'user_id' in session:
        posts = Post.query.order_by(Post.timestamp.desc()).all()
        return render_template('home.html', posts=posts)
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    """Display posts specific to the logged-in user."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    posts = Post.query.filter_by(user_id=user_id).order_by(Post.timestamp.desc()).all()
    return render_template('profile.html', posts=posts)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/create_post', methods=['GET', 'POST'])
def create_post():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        post = Post(title=title, content=content, user_id=session['user_id'])
        db.session.add(post)
        db.session.commit()
        flash('Post created successfully.', 'success')
        return redirect(url_for('profile'))
    return render_template('create_post.html')

@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def post_details(post_id):
    post = Post.query.get_or_404(post_id)
    harassment_alert = None  # Initialize harassment alert

    if request.method == 'POST':
        content = request.form['content']
        # Preprocess the comment content
        cleaned_content = preprocess_input(content)

        # Predict harassment type
        harassment_type = predict_harassment_type(content)

        # Tokenize and pad the text
        seq = tokenizer.texts_to_sequences([cleaned_content])
        padded_seq = pad_sequences(seq, maxlen=50, padding='post', truncating='post')

        # Predict class using XGBoost
        predicted_class = xgboost_model.predict(np.array(padded_seq))[0]

        # Predict severity using LSTM
        predicted_severity_numeric = severity_model.predict(padded_seq)[0][0]
        predicted_severity = severity_mapping.get(round(predicted_severity_numeric), "Unknown")

        # Check if harassment is detected
        if harassment_type != "None":
            # Populate harassment alert with details
            harassment_alert = {
                "harassment_type": harassment_type,
                "predicted_class": predicted_class,
                "predicted_severity": predicted_severity
            }
        print(harassment_type,"=======",predicted_class,"=========",predicted_severity)
        # Save the comment
        comment = Comment(content=content, post_id=post.id)
        db.session.add(comment)
        db.session.commit()
        flash('Comment added!', 'success')

    return render_template(
        'post_details.html',
        post=post,
        harassment_alert=harassment_alert
    )


# Initialize Database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
