# Telecom Churn Prediction

## Introduction and Problem Statement

### Business Problem Overview

In the telecom industry, customers can choose from multiple service providers and actively switch from one operator to another. This results in an average annual churn rate of **15-25%**. Given that acquiring a new customer costs **5-10 times** more than retaining an existing one, customer retention has become a higher priority than customer acquisition.

For many telecom operators, **retaining highly profitable customers is the number one business goal**. To minimize churn, telecom companies need to predict which high-value customers are at risk and take proactive measures.

### Objective

This project analyzes **customer-level data** from a leading telecom firm to:
- **Build predictive models** to identify high-risk customers.
- **Identify key churn indicators** to help businesses retain customers.

## Understanding and Defining Churn

### Telecom Business Models

Telecom services operate under two primary models:
1. **Postpaid Model** - Customers receive a bill for services used and must notify the provider to switch operators.
2. **Prepaid Model** - Customers pay in advance and can stop using services without notice, making churn prediction challenging.

Churn prediction is especially **critical in prepaid services**, which dominate markets in **India and Southeast Asia**.

### Defining Churn

There are multiple ways to define churn:
1. **Revenue-based churn** - Customers who do not generate revenue for a given period.
2. **Usage-based churn** - Customers with zero incoming or outgoing calls, SMS, or mobile data usage for a defined time.

This project uses the **usage-based churn definition**.

## High-Value Customer Churn

Approximately **80% of telecom revenue** comes from the **top 20% of customers** (high-value customers). Reducing churn among these customers minimizes revenue loss significantly.

### Identifying High-Value Customers

Customers who have recharged with an amount **greater than or equal to the 70th percentile of the average recharge amount** in the first two months are classified as high-value customers.

## Dataset Overview

### Data Collection

The dataset contains **customer-level information over four months** (June-September), labeled as months **6, 7, 8, and 9**.

### Business Objective

The goal is to **predict churn in month 9** using features from months **6, 7, and 8**.

### Customer Behavior Phases

- **Good Phase**: Customers are satisfied (Months 6 & 7)
- **Action Phase**: Customers start experiencing dissatisfaction (Month 8)
- **Churn Phase**: Customers have churned (Month 9)

After defining churn, **all data related to month 9 is removed from modeling**.

### Data Dictionary

The dataset includes various features such as:
- **Recharge amounts** (total_rech_amt_6, total_rech_amt_7, etc.)
- **Usage data** (total_ic_mou_6, total_og_mou_6, vol_2g_mb_6, etc.)
- **Customer interactions** (calls, SMS, data usage)

## Data Preparation

### Steps Taken

1. **Filtering High-Value Customers**
   - Customers who have recharged above the **70th percentile** of average recharge amounts in months **6 and 7** are considered.

2. **Tagging Churners**
   - Customers with **zero incoming/outgoing calls and zero mobile data usage in month 9** are marked as churners.

3. **Dropping Month 9 Data**
   - Since month 9 represents churn, all related columns are removed.

## Model Development

### Predictive Modeling Goals

1. **Predict churn probability for high-value customers**.
2. **Identify the most important features driving churn**.

### Handling Class Imbalance

Since churn rates are typically **low (5-10%)**, techniques such as **oversampling, undersampling, or SMOTE** are considered.

### Model Selection

The following machine learning models are implemented:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

## Model Evaluation

### Evaluation Metrics

- **Accuracy**: Measures correct predictions.
- **ROC-AUC Score**: Evaluates how well the model distinguishes churners vs. non-churners.
- **Confusion Matrix**: Helps analyze false positives and false negatives.
- **Feature Importance**: Identifies top features influencing churn.

## Results

- **XGBoost performed the best**, achieving the highest ROC-AUC score.
- **Feature Importance Analysis** revealed key churn indicators:
  - **Recharge Amounts**: Drop in recharge amounts signals churn risk.
  - **Call Duration**: Lower call minutes indicate customer dissatisfaction.
  - **Data Usage**: Reduced internet usage suggests potential churn.

## Business Recommendations

### **Preventive Measures to Reduce Churn**

1. **Early Identification of At-Risk Customers**
   - Offer personalized retention plans to customers showing reduced activity.

2. **Customer-Centric Retention Strategies**
   - Special discounts, loyalty programs, and exclusive plans for high-value customers.

3. **Service Quality Improvement**
   - Enhancing network infrastructure and customer support to improve user experience.

4. **Competitive Analysis**
   - Monitor competitor offerings and proactively adjust pricing and services.

## Conclusion

This project successfully developed a predictive model to **identify high-risk churners**, allowing telecom companies to take proactive retention measures. By implementing these insights, businesses can significantly reduce churn and enhance customer loyalty.

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


