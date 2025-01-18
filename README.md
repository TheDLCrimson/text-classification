# Project ML: Comparing Gemini Flash and Flash8B Models

## Overview

This project implements a machine learning pipeline to compare the performance of two text classification models: **Gemini Flash** and **Gemini Flash8B**. Both models classify text data into three categories:

- `-1` (UNCLEAR): Text that lacks clarity or is ambiguous.
- `0` (REJECT): Text that should be flagged or rejected.
- `1` (BYPASS): Text that is acceptable and passes the classification criteria.

The pipeline processes input text, classifies the responses using both models, and generates insights into their performance through visualizations and summary statistics.

## Purpose

The primary objective of this project is to compare the classification results of the Gemini Flash and Flash8B models. By analyzing the distribution of predictions across various datasets, users can:

1. Understand the strengths and weaknesses of each model.
2. Identify trends in classification performance.
3. Make informed decisions on which model to use for specific applications.

## Workflow

### 1. **Dataset Preparation**

- Input files are stored in two folders:
  - `gemini_flash_response`: Contains text responses to be classified by the Gemini Flash model.
  - `gemini_flash-8b_response`: Contains text responses to be classified by the Gemini Flash8B model.
- Each folder contains multiple CSV files, one for each language, with a column named `response_english` holding the text data to classify.

### 2. **Text Preprocessing**

To standardize the input data, the following preprocessing steps are applied:

- Special characters and digits are removed using regular expressions.
- Text is converted to lowercase and tokenized into individual words.
- Common stopwords (e.g., "the", "and", "is") are removed using the NLTK library.
- Words are stemmed using the Porter Stemmer to reduce them to their root forms.

### 3. **Classification**

- Both models (Gemini Flash and Gemini Flash8B) are trained on separate datasets and saved as `best_model.pkl`.
- A pre-trained TF-IDF vectorizer (`vectorizer.pkl`) is used to convert text into numerical features.
- For each folder, the responses are classified, and the results are saved:
  - Classified files are stored in a `classified` subfolder within each response folder.
  - A summary file containing the distribution of labels (`-1`, `0`, `1`) for each language is saved as:
    - `classification_flash_results_summary.csv` (for Gemini Flash)
    - `classification_flash8b_results_summary.csv` (for Gemini Flash8B)

### 4. **Visualization and Comparison**

- Results for both models are visualized using stacked bar charts:
  - Each language is represented along the x-axis.
  - Bars show the count and percentage of predictions for each class (`-1`, `0`, `1`).
  - Separate graphs are generated for Gemini Flash and Gemini Flash8B.
- The charts provide an intuitive comparison of how the two models classify text across languages.

## Files

- **Model and Vectorizer**:
  - `best_model.pkl`: The trained classification model.
  - `vectorizer.pkl`: The TF-IDF vectorizer used for text preprocessing.
- **Input Files**:
  - `gemini_flash_response/`: Contains text responses for the Gemini Flash model.
  - `gemini_flash-8b_response/`: Contains text responses for the Gemini Flash8B model.
- **Output Files**:
  - `classified/`: Subfolder containing classified responses for each language.
  - `classification_flash_results_summary.csv`: Summary of results for Gemini Flash.
  - `classification_flash8b_results_summary.csv`: Summary of results for Gemini Flash8B.

## How to Run

### Prerequisites

1. Install required dependencies:
   ```
   pip install numpy pandas scikit-learn nltk matplotlib joblib
   ```
2. Ensure that input files are organized into the folders `gemini_flash_response/` and `gemini_flash-8b_response/`.

### Steps

1. **Classify Responses**:
   - Run the classification script for both models:
     ```
     python classify_flash.py  # For Gemini Flash
     python classify_flash8b.py  # For Gemini Flash8B
     ```
2. **Generate Visualizations**:
   - Run the visualization script to compare results:
     ```
     python compare_graph.py
     ```
3. **Analyze Results**:
   - Review the classification summaries (`classification_flash_results_summary.csv` and `classification_flash8b_results_summary.csv`).
   - Compare the stacked bar charts to identify differences in performance between the models.

## Results

- The project generates classified response files for each model.
- Summaries provide counts and percentages of each class (`-1`, `0`, `1`) by language.
- Visualizations highlight differences in classification patterns between Gemini Flash and Flash8B.

## Insights

Through this project, users can:

1. Identify languages or datasets where one model outperforms the other.
2. Evaluate consistency in predictions across different languages.
3. Make data-driven decisions on which model to deploy for specific applications.

This comparative analysis ensures that the best-performing model is chosen for the desired use case, improving overall classification accuracy and reliability.
