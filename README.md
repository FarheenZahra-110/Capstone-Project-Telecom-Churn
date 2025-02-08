# Telecom Churn Prediction

## Project Overview
This project aims to predict customer churn in the telecom sector using machine learning models. The dataset contains customer usage patterns, and the goal is to identify high-value customers who are likely to churn in the future. 

## Dataset
- The dataset contains customer recharge amounts, call durations, and data usage over multiple months.
- Churn is defined as customers who have not used incoming or outgoing calls and data services in the last observed month.

## Steps Involved
1. **Data Preprocessing**
   - Handle missing values and convert columns to appropriate formats.
   - Select high-value customers based on recharge amounts.
   - Define churn based on customer inactivity in the last month.
   - Drop churn-phase columns to prevent data leakage.

2. **Feature Engineering & Scaling**
   - Standardize numerical features using `StandardScaler`.
   - Select appropriate features for training.

3. **Model Training & Evaluation**
   - Train three classification models: Logistic Regression, Random Forest, and XGBoost.
   - Evaluate models using accuracy, AUC-ROC score, and classification report.
   - Visualize ROC curves for comparison.

4. **Feature Importance Analysis**
   - Identify top features influencing churn prediction using the best-performing model.
   - Visualize feature importance.

5. **Business Recommendations**
   - Target high-risk customers with special offers.
   - Improve network quality in high-churn areas.
   - Provide loyalty benefits to early churn indicators.
  
   - 
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

## Author
- **Farheen Zahra**
