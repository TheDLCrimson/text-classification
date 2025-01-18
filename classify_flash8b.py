import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.lower().split()  # Convert to lowercase and tokenize
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]  # Apply stemming
    return ' '.join(text)

# Load the best model and vectorizer
best_model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to preprocess and classify responses
def classify(file_path, vectorizer, model):
    # Load the response file
    try:
        response_data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")
    
    if 'response_english' not in response_data.columns:
        raise ValueError(f"Column 'response_english' not found in file: {file_path}")
    
    # Preprocess the text
    response_data['response_english'] = response_data['response_english'].apply(preprocess)
    
    # Vectorize the text
    vectorized_responses = vectorizer.transform(response_data['response_english'])
    
    # Predict labels
    response_data['label'] = model.predict(vectorized_responses)
    
    # Count label occurrences
    label_counts = response_data['label'].value_counts().to_dict()
    
    # Define output directory and save classified responses
    output_dir = "gemini_flash8b_response/classified"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(file_path).replace(".csv", "_classified.csv"))
    response_data.to_csv(output_file, index=False)
    
    return label_counts, output_file

# List of uploaded files in the gemini_flash8b_response folder
file_paths = [
    "gemini_flash8b_response/english_ar_respone.csv",
    "gemini_flash8b_response/english_bn_respone.csv",
    "gemini_flash8b_response/english_en_respone.csv",
    "gemini_flash8b_response/english_gd_respone.csv",
    "gemini_flash8b_response/english_gn_respone.csv",
    "gemini_flash8b_response/english_he_respone.csv",
    "gemini_flash8b_response/english_hi_respone.csv",
    "gemini_flash8b_response/english_hmn_respone.csv",
    "gemini_flash8b_response/english_it_respone.csv",
    "gemini_flash8b_response/english_th_respone.csv",
    "gemini_flash8b_response/english_uk_respone.csv",
    "gemini_flash8b_response/english_zh-cn_respone.csv",
    "gemini_flash8b_response/english_zu_respone.csv",
]

# Dictionary to store results
results = {}

# Process each file
for file_path in file_paths:
    try:
        counts, output_file = classify(file_path, vectorizer, best_model)
        language = os.path.basename(file_path).split('_')[1]  # Extract language code
        results[language] = counts
        print(f"Classified file saved to: {output_file}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Save the classification results summary to a CSV file
results_df = pd.DataFrame.from_dict(results, orient="index").fillna(0).astype(int)
results_df = results_df.rename_axis("Language").reset_index()
output_results_file = "classification_flash8b_results_summary.csv"
results_df.to_csv(output_results_file, index=False)

output_results_file

