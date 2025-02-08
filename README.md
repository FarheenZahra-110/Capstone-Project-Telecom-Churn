# Telecom Customer Churn Prediction

## Overview
This project aims to predict customer churn in the telecom industry by analyzing user behavior and identifying high-risk customers. The goal is to help the company take proactive measures to retain customers by offering personalized services and benefits.

## Dataset Information
The dataset contains telecom user details, including recharge amounts, call durations, and data usage over multiple months. Churn is defined based on customers who show no activity in the last month.

### Key Features:
- **Recharge Amounts:** Monthly recharge amounts for different periods.
- **Call Usage:** Incoming and outgoing call minutes.
- **Data Usage:** 2G and 3G data consumption.
- **Churn Indicator:** Defined as users with no outgoing/incoming calls and no data usage in the last observed month.

## Implementation Steps
### 1. Data Preprocessing
- Load the dataset and perform initial exploration.
- Filter high-value customers based on recharge amount thresholds.
- Handle missing values and data inconsistencies.

### 2. Feature Engineering
- Identify the most relevant features impacting customer churn.
- Remove data from the churn phase to avoid data leakage.
- Standardize numerical variables for better model performance.

### 3. Model Training & Evaluation
Three classification models were trained and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

#### Model Evaluation Metrics:
- **Accuracy Score**
- **AUC-ROC Score**
- **Confusion Matrix & Classification Report**

### 4. Feature Importance Analysis
- The best-performing model was selected based on AUC-ROC score.
- Important features contributing to churn prediction were identified using feature importance from Random Forest/XGBoost.

## Results
- **Best Model:** Random Forest/XGBoost (whichever had the highest AUC-ROC score)
- **Top Contributing Features:**
  - Total recharge amount
  - Outgoing call duration
  - Data consumption trends
  
## Business Recommendations
- **Target high-risk customers** with special offers before they churn.
- **Improve network quality** in areas with high churn rates.
- **Offer loyalty benefits** to users showing early churn behavior.
- **Enhance customer engagement** by analyzing top churn predictors and designing personalized retention strategies.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix

# Load dataset
df = pd.read_csv('/mnt/data/telecom_churn_data.csv')
print("Dataset Loaded Successfully")

# Display basic info
print("Dataset Info:")
df.info()
print("\nDataset Head:")
print(df.head())

# Filter High-Value Customers
X_thresh = df[['total_rech_amt_6', 'total_rech_amt_7']].mean(axis=1).quantile(0.7)
df_high_value = df[df[['total_rech_amt_6', 'total_rech_amt_7']].mean(axis=1) >= X_thresh]

# Tag Churners
df_high_value['churn'] = np.where((df_high_value['total_ic_mou_9'] == 0) &
                                   (df_high_value['total_og_mou_9'] == 0) &
                                   (df_high_value['vol_2g_mb_9'] == 0) &
                                   (df_high_value['vol_3g_mb_9'] == 0), 1, 0)

# Drop Churn Phase Data
df_high_value = df_high_value.drop(columns=[col for col in df_high_value.columns if '_9' in col])

# Define Features and Target
X = df_high_value.drop(columns=['churn'])
y = df_high_value['churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize Numerical Features
num_cols = X_train.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Model Training & Evaluation
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.2f}')

    return auc_roc

plt.figure(figsize=(8, 6))

# Train Logistic Regression
log_reg = LogisticRegression()
auc_lr = train_and_evaluate(log_reg, "Logistic Regression")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
auc_rf = train_and_evaluate(rf, "Random Forest")

# Train XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
auc_xgb = train_and_evaluate(xgb, "XGBoost")

# Finalize and Show ROC Curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Feature Importance Analysis
best_model = rf if auc_rf > auc_xgb else xgb
feature_importance = best_model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)
print("\nTop Features:")
print(feat_imp_df.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp_df['Importance'][:10], y=feat_imp_df['Feature'][:10])
plt.title("Top 10 Important Features")
plt.show()

# Business Recommendations
print("\n--- Business Recommendations ---")
print("- Target high-risk customers with special offers before churn.")
print("- Improve network quality in regions where churn is high.")
print("- Offer loyalty benefits to customers showing early churn behavior.")

## Conclusion
This project provides valuable insights into telecom customer churn patterns and suggests actionable strategies for retention. By leveraging machine learning, businesses can predict churn early and take necessary steps to improve customer loyalty and revenue retention.


