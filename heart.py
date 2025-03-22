import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from experta import *

# ---------------------------
# Step 1: Dataset Processing
# ---------------------------
# Load dataset
df = pd.read_csv("heart (2).csv")
# Handle missing values (choose to fill with median)
df.fillna(df.median(), inplace=True)

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Encode categorical variables using One-Hot Encoding
categorical_features = ["cp", "restecg", "slope", "thal", "exang"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Save processed data
df.to_csv("cleaned_data.csv", index=False)

# ---------------------------
# Step 2: Data Visualization
# ---------------------------
# Statistical Summary
df.describe().to_csv("statistical_summary.csv")  # Save summary

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")  # Save figure
plt.show()

# Histograms of features
df.hist(figsize=(12, 8), bins=20)
plt.savefig("histograms.png")  # Save figure
plt.show()

# Feature importance using Decision Tree
X = df.drop(columns=["target"])
y = df["target"]
model = DecisionTreeClassifier()
model.fit(X, y)

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind="barh", title="Feature Importance")
plt.savefig("feature_importance.png")  # Save figure
plt.show()

# ---------------------------
# Step 3: Rule-Based Expert System (Experta)
# ---------------------------
class Patient(Fact):
    """Patient attributes"""
    pass

class HeartRiskExpert(KnowledgeEngine):
    def _init_(self):
        super()._init_()
        self.result = 0  # Default risk level

    # Define exactly 10 rules
    @Rule(Patient(chol=P(lambda x: x > 0.8)) & Patient(age=P(lambda x: x > 0.6)))
    def high_risk_1(self):
        self.result = 1

    @Rule(Patient(trestbps=P(lambda x: x > 0.7)) & Patient(exang_1=1))
    def high_risk_2(self):
        self.result = 1

    @Rule(Patient(thalach=P(lambda x: x > 0.9)) & Patient(oldpeak=P(lambda x: x < 0.2)))
    def low_risk(self):
        self.result = 0

    @Rule(Patient(trestbps=P(lambda x: x > 0.8)) & Patient(age=P(lambda x: x > 0.5)))
    def high_risk_3(self):
        self.result = 1

    @Rule(Patient(chol=P(lambda x: x > 0.9)) & Patient(oldpeak=P(lambda x: x > 0.3)))
    def high_risk_4(self):
        self.result = 1

    @Rule(Patient(thalach=P(lambda x: x < 0.3)) & Patient(age=P(lambda x: x > 0.7)))
    def high_risk_5(self):
        self.result = 1

    @Rule(Patient(exang_1=0) & Patient(thalach=P(lambda x: x > 0.8)))
    def low_risk_2(self):
        self.result = 0

    @Rule(Patient(chol=P(lambda x: x > 0.85)) & Patient(age=P(lambda x: x > 0.7)))
    def high_risk_6(self):
        self.result = 1

    @Rule(Patient(trestbps=P(lambda x: x > 0.75)) & Patient(thalach=P(lambda x: x < 0.5)))
    def high_risk_7(self):
        self.result = 1

    @Rule(Patient(oldpeak=P(lambda x: x > 0.4)) & Patient(age=P(lambda x: x > 0.6)))
    def high_risk_8(self):
        self.result = 1

    @Rule(Patient(cp_1=1) & Patient(exang_1=1))
    def high_risk_9(self):
        self.result = 1

def apply_expert_system(row):
    engine = HeartRiskExpert()
    engine.reset()
    engine.declare(
        Patient(chol=row["chol"]),
        Patient(age=row["age"]),
        Patient(trestbps=row["trestbps"]),
        Patient(exang_1=row.get("exang_1", 0)),
        Patient(thalach=row["thalach"]),
        Patient(oldpeak=row["oldpeak"])
    )
    engine.run()
    
    return getattr(engine, "result", 0)

df["expert_prediction"] = df.apply(apply_expert_system, axis=1)
# Step 4: Build Decision Tree Model
# ---------------------------
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
tree_model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
tree_model.fit(X_train, y_train)
joblib.dump(tree_model, "decision_tree_model.pkl")  # Save model
# Make predictions and evaluate
y_pred_tree = tree_model.predict(X_test)
tree_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_tree),
    "Precision": precision_score(y_test, y_pred_tree),
    "Recall": recall_score(y_test, y_pred_tree),
    "F1-Score": f1_score(y_test, y_pred_tree)
}

# Evaluate Expert System
expert_metrics = {
    "Accuracy": accuracy_score(y_test, df.loc[y_test.index, "expert_prediction"]),
    "Precision": precision_score(y_test, df.loc[y_test.index, "expert_prediction"]),
    "Recall": recall_score(y_test, df.loc[y_test.index, "expert_prediction"]),
    "F1-Score": f1_score(y_test, df.loc[y_test.index, "expert_prediction"])
}

# Print performance comparison
comparison_results = pd.DataFrame({
    "Model": ["Expert System", "Decision Tree"],
    "Accuracy": [expert_metrics["Accuracy"], tree_metrics["Accuracy"]],
    "Precision": [expert_metrics["Precision"], tree_metrics["Precision"]],
    "Recall": [expert_metrics["Recall"], tree_metrics["Recall"]],
    "F1-Score": [expert_metrics["F1-Score"], tree_metrics["F1-Score"]]
})

print("\nPerformance Comparison:")
print(comparison_results)
# ---------------------------
# Step 5: User Input Prediction
# ---------------------------
def predict_user_input():
    print("\nEnter your details for heart disease risk prediction:")
    user_data = {
        "age": float(input("Age (normalized 0-1): ")),
        "trestbps": float(input("Blood Pressure (normalized 0-1): ")),
        "chol": float(input("Cholesterol (normalized 0-1): ")),
        "thalach": float(input("Max Heart Rate (normalized 0-1): ")),
        "oldpeak": float(input("ST Depression (normalized 0-1): ")),
        "exang_1": int(input("Exercise Induced Angina (1: Yes, 0: No): "))
    }
# Load trained model and make prediction
    tree_model = joblib.load("decision_tree_model.pkl")
    user_df = pd.DataFrame([user_data])
     # Ensure columns match the model's input
    missing_cols = [col for col in X.columns if col not in user_df.columns]
    for col in missing_cols:
        user_df[col] = 0  # Set missing categorical columns to 0

    tree_prediction = tree_model.predict(user_df)[0]

 # Expert System Prediction
    expert_prediction = apply_expert_system(user_df.iloc[0])

    print("\nPrediction Result:")
    print(f"Decision Tree Prediction: {'Heart Disease' if tree_prediction == 1 else 'No Heart Disease'}")
    print(f"Expert System Prediction: {'Heart Disease' if expert_prediction == 1 else 'No Heart Disease'}")

# Uncomment this line to test user input prediction
# predict_user_input()
