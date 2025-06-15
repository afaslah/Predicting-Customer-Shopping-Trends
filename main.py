# Import required libraries
from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#TITLE
st.set_page_config(page_title="Shopping Trends Predictor", layout="wide")

def load_and_preprocess_data(data):
    purchase_frequency_map = {
        'Weekly': 1, 'Bi-Weekly': 1, 'Fortnightly': 1,
        'Monthly': 0, 'Quarterly': 0, 'Every 3 Months': 0, 'Annually': 0
    }
    data['Frequent_Buyer'] = data['Frequency of Purchases'].map(purchase_frequency_map)
    data['Discount_Seeker'] = data['Discount Applied'].map({'Yes': 1, 'No': 0})
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Category', 'Location', 'Size', 'Color', 'Season',
                           'Subscription Status', 'Shipping Type', 'Payment Method']
    for col in categorical_columns:
        data[col + '_Encoded'] = le.fit_transform(data[col])
    return data

# Prepare features
def prepare_features(data, target):
    feature_columns = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases',
                       'Gender_Encoded', 'Category_Encoded', 'Location_Encoded', 'Size_Encoded',
                       'Color_Encoded', 'Season_Encoded', 'Subscription Status_Encoded',
                       'Shipping Type_Encoded', 'Payment Method_Encoded']
    X = data[feature_columns]
    y = data[target]
    return X, y

# Train model
def train_models(X_train, X_test, y_train, y_test):
    
    # Log Reg
    lr_model = LogisticRegression(random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]

    # Hutan
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    # evaluation
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr_proba)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)


    metrics = {
        "Logistic Regression": {
            "Accuracy": lr_accuracy,
            "AUC": lr_auc,
            "F1 Score": lr_f1,
            "Precision": lr_precision,
            "Recall": lr_recall
        },
        "Random Forest": {
            "Accuracy": rf_accuracy,
            "AUC": rf_auc,
            "F1 Score": rf_f1,
            "Precision": rf_precision,
            "Recall": rf_recall
        }
    }

    return (lr_model, rf_model), metrics, (lr_pred, rf_pred)

# Confusion Matrix
def plot_confusion_matrix(y_test, predictions, model_name, ax):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


# Main Streamlit app
def main():
    st.title("Shopping Trends Prediction Dashboard")

    data = pd.read_csv('shopping_trends_updated.csv')
    processed_data = load_and_preprocess_data(data)
    
    st.sidebar.header("Model Settings")
    prediction_task = st.sidebar.selectbox(
        "Select Prediction Task",
        ["Frequent Buyer Prediction", "Discount Seeker Prediction"]
    )
    
    if prediction_task == "Frequent Buyer Prediction":
        X, y = prepare_features(processed_data, 'Frequent_Buyer')
        title = "Frequent Buyer"
    else:
        X, y = prepare_features(processed_data, 'Discount_Seeker')
        title = "Discount Seeker"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models, metrics, predictions = train_models(X_train, X_test, y_train, y_test)
    lr_model, rf_model = models
    lr_pred, rf_pred = predictions

    st.header(f"{title} Prediction Results")

    # Display model evaluation in a table with all metrics
    st.subheader("Model Evaluation")
    evaluation_df = pd.DataFrame(metrics).T  # Transpose to display models as rows
    evaluation_df = evaluation_df.applymap(lambda x: f"{x:.2%}" if isinstance(x, float) else x)
    st.table(evaluation_df)

    # Display Confusion Matrices
    st.subheader("Confusion Matrix")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", axs[0])
    plot_confusion_matrix(y_test, rf_pred, "Random Forest", axs[1])
    st.pyplot(fig)

if __name__ == "__main__":
    main()
