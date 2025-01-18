import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import joblib

nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower().split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return ' '.join(text)

# Load and preprocess training data
train_data = pd.read_csv('train.csv')
train_data['target'] = train_data['target'].apply(preprocess)

y_train = train_data['label'].tolist()
x_train = train_data['target']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

# Model definitions
models = {
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=50, min_samples_split=5, min_samples_leaf=2, random_state=1),
    "Logistic Regression": LogisticRegression(max_iter=500, penalty='l2', solver='liblinear', class_weight="balanced", random_state=1),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
}

# Train and evaluate models
best_model = None
best_f1_score = 0
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    y_val_pred = model.predict(X_val_vectorized)
    print(f"Performance of {name} on Validation Set:")
    report = classification_report(y_val, y_val_pred, output_dict=True)
    print(classification_report(y_val, y_val_pred))
    print("-" * 50)
    if report['weighted avg']['f1-score'] > best_f1_score:
        best_model = model
        best_f1_score = report['weighted avg']['f1-score']

# Save the best model and vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully.")
# Load test data and make predictions
def load_predict(file, vectorizer, model):
    test_data = pd.read_csv(file)
    test_data['target'] = test_data['target'].apply(preprocess)  # Preprocess test data
    vectorized_x = vectorizer.transform(test_data['target'])
    test_data['label'] = model.predict(vectorized_x)
    return test_data[['target', 'label']]

# Load and predict using the best model
predictions = load_predict('test.csv', vectorizer, best_model)

# Save predictions to CSV
predictions.to_csv('results.csv', index=False)
print("Predictions saved to 'results.csv'")
