from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load data from file
def load_data(file):
    df = pd.read_csv(file)
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        df = load_data(file)
        return analyze_data(df)
    return redirect(url_for('index'))

@app.route('/manual', methods=['POST'])
def manual_data():
    data = request.form['data']
    df = pd.read_csv(io.StringIO(data))
    return analyze_data(df)

def analyze_data(df):
    # Check if the dataset has the required columns for Titanic dataset
    required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
    if not all(col in df.columns for col in required_columns):
        return "Dataset must contain the following columns: " + ", ".join(required_columns)
    
    # Preprocessing: Convert categorical columns to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Handle missing values
    df['Fare'].fillna(df['Fare'].median(), inplace=True)  # Handle missing values
    
    # Feature matrix and target variable
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    # Generate a histogram of predictions
    plt.figure()
    plt.hist(y_pred, bins=2, rwidth=0.8, color='blue', alpha=0.7)
    plt.title('Prediction Histogram')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    
    # Save plot to a buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf8')
    
    return render_template('results.html', report=report, img=img)

if __name__ == '__main__':
    app.run(debug=True)