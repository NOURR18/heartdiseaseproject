This project aims to predict the risk of heart disease using two models: a *Decision Tree Classifier* and a *Rule-Based Expert System*. The models are built based on various medical parameters and use a dataset of heart disease records.

## Project Structure

1. *Dataset Processing*:
   - Loading the dataset, handling missing values, normalizing numerical features, and encoding categorical variables using one-hot encoding.
   
2. *Data Visualization*:
   - Generating statistical summaries and visualizations such as heatmaps, histograms, and feature importance using the Decision Tree model
 
3. *Rule-Based Expert System (Experta)*:
   - Implementing a knowledge-based expert system using experta library to predict heart disease risk based on predefined rules.

4. *Model Building*:
   - Training a Decision Tree Classifier and evaluating its performance using metrics like accuracy, precision, recall, and F1-score.

5. *User Input Prediction*:
   - Allowing users to input their medical data for prediction using both the Decision Tree model and the Expert System.

## Files in the Repository
	•	heart.py: The main Python script containing all the steps for dataset processing, model training, and user prediction.
	•	cleaned_data.csv: The processed dataset after handling missing values and normalizing features.
	•	statistical_summary.csv: A summary of statistical details about the dataset.
	•	correlation_heatmap.png: A visual representation of feature correlations.
	•	histograms.png: Histograms of different features.
	•	feature_importance.png: A bar chart showing the most important features based on the Decision Tree.
	•	decision_tree_model.pkl: The saved Decision Tree model used for predictions.
