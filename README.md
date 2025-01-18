# README: Classification Pipeline for Text Data

## Overview
This project implements a machine learning pipeline to classify text data into three categories: `-1` (UNCLEAR), `0` (REJECT), and `1` (BYPASS). The pipeline processes input data, handles class imbalances, trains multiple models, and evaluates their performance to identify the best-performing model for text classification tasks.

## Workflow

### 1. **Dataset Preparation**
- The training data (`train.csv`) contains two columns:
  - `target`: The text data to be classified.
  - `label`: The ground truth labels for classification (`-1`, `0`, `1`).
- The dataset is preprocessed to clean and standardize the text before being split into training and validation sets.

### 2. **Text Preprocessing**
- Special characters and digits are removed using regular expressions.
- Text is converted to lowercase and tokenized into words.
- Stopwords are removed using the NLTK library.
- Words are reduced to their root forms using stemming (`PorterStemmer`).

### 3. **TF-IDF Vectorization**
- Text data is transformed into numerical features using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
- Bi-gram features are extracted (`ngram_range=(1, 2)`), and the number of features is limited to 5000 for efficiency.

### 4. **Handling Class Imbalance**
- Class imbalances in the training data are mitigated using **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for underrepresented classes.

### 5. **Model Training and Validation**
Four machine learning models are trained and evaluated:
- **K-Nearest Neighbors (K-NN)**
- **Decision Tree**
- **Logistic Regression**
- **Neural Network (MLPClassifier)**

Each model is trained on the resampled training data and evaluated on the validation set using metrics:
- Precision
- Recall
- F1-score
- Accuracy

The best-performing model is selected based on the weighted average F1-score.

### 6. **Test Predictions**
- A test dataset (`test.csv`) is preprocessed similarly to the training data.
- The best model is used to predict labels for the test dataset.
- The output contains two columns:
  - `target`: The preprocessed text data.
  - `label`: The predicted class for each text sample.
- Predictions are saved to `results.csv`.

### 7. **Model Saving**
- The best-performing model and the TF-IDF vectorizer are saved using `joblib` for reuse in future predictions.

## Files
- `train.csv`: Training dataset containing `target` and `label` columns.
- `test.csv`: Test dataset to evaluate the model on unseen data.
- `results.csv`: Output file containing the predicted labels for the test dataset.
- `best_model.pkl`: Saved best model.
- `vectorizer.pkl`: Saved TF-IDF vectorizer.

## Results
The models achieved perfect scores (precision, recall, F1-score, and accuracy) on the validation set. However, further steps, such as using a more complex test set or performing cross-validation, are recommended to ensure robustness and avoid overfitting.

## How to Run
1. Install required dependencies:
   ```
   pip install numpy pandas scikit-learn imbalanced-learn nltk joblib
   ```
2. Run the Python script to train models and generate predictions:
   ```
   python main.py
   ```
3. Check `results.csv` for the classification results on the test dataset.

