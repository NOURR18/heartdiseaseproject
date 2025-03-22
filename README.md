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
   - 
# #Using the Model:
•The system will prompt you to enter your details (age, cholesterol levels, blood pressure, etc.). The program will output the predicted risk of heart disease from both the Decision Tree Model and the Expert System.
## Viewing Results:
You can also view the statistical summary, heatmap, and other visualizations generated during data analysis

## Files in the Repository
	•	heart.py: The main Python script containing all the steps for dataset processing, model training, and user prediction.
	•	cleaned_data.csv: The processed dataset after handling missing values and normalizing features.
	•	statistical_summary.csv: A summary of statistical details about the dataset.
	•	correlation_heatmap.png: A visual representation of feature correlations.
	•	histograms.png: Histograms of different features.
	•	feature_importance.png: A bar chart showing the most important features based on the Decision Tree.
	•	decision_tree_model.pkl: The saved Decision Tree model used for predictions.
 
## Example Inputs

For user input prediction, enter values as instructed:
	•Age (normalized 0-1)
	•Cholesterol (normalized 0-1)
	•Blood Pressure (normalized 0-1)
	•max Heart Rate (normalized 0-1)
	•ST Depression (normalized 0-1)
	•Exercise-Induced Angina (1: Yes, 0: No)
 
 ## Model Evaluation

The project evaluates the models using metrics like:
	•Accuracy: The proportion of correct predictions.
	•Precision: The proportion of positive predictions that are correct.
	•Recall: The proportion of actual positives correctly predicted.
	•F1-Score: A balance between precision and recall.
 ## how to run the project 
 After setting up the environment,you can run the main script (heart.py)
