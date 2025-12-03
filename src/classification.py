# Classification
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier  # Import KNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed data and TF-IDF matrix from preprocessing_tfidf.py
df = pd.read_excel('C:\\Users\\Safiyyah\\eclipse-workspace\\processed_ASSIGN-ECOMMERCE WOMAN CLOTHING.xlsx')
tfidf_df = pd.read_csv('C:\\Users\\Safiyyah\\eclipse-workspace\\TFIDF.csv', index_col=0)

# Set up target variable and feature matrix
y = df['RECOMMENDED IND']
X_tfidf = tfidf_df.to_numpy()

# Step 1: Feature Selection with Chi-Square
print("\nStarting feature selection using Chi-Square...")
chi2_selector = SelectKBest(chi2, k=1000)
X_selected = chi2_selector.fit_transform(X_tfidf, y)
print("Feature selection completed.")

# Print the top 20 most informative features based on Chi-Square test
chi2_scores = chi2_selector.scores_
top_k_indices = np.argsort(chi2_scores)[::-1][:20]
top_k_features = [tfidf_df.columns[i] for i in top_k_indices]
print("\nTop 20 Features based on Chi-Square Test:")
for feature in top_k_features:
    print(feature)

# Step 2: Model Specification and Initialization
# Define models
    #Model Based- Generative
nb_model = MultinomialNB()
    #Model Based- Discriminative
svm_model = LinearSVC(max_iter=10000, random_state=42)
    #Model- Less
knn_model = KNeighborsClassifier(n_neighbors=5)  

# Step 3: Model Estimation and Cross-Validation
kfold = StratifiedKFold(n_splits=5)

# Function to perform cross-validation and print performance metrics
def evaluate_model(model, X, y):
    accuracy = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=kfold, scoring='precision').mean()
    recall = cross_val_score(model, X, y, cv=kfold, scoring='recall').mean()
    f1 = cross_val_score(model, X, y, cv=kfold, scoring='f1').mean()
    return accuracy, precision, recall, f1

# Evaluate Naive Bayes
print("\nEvaluating Naive Bayes model...")
nb_accuracy, nb_precision, nb_recall, nb_f1 = evaluate_model(nb_model, X_selected, y)

print("\nNaive Bayes Cross-Validation Results:")
print(f"Accuracy: {nb_accuracy:.4f}")
print(f"Precision: {nb_precision:.4f}")
print(f"Recall: {nb_recall:.4f}")
print(f"F1 Score: {nb_f1:.4f}")

# Evaluate LinearSVC
print("\nEvaluating LinearSVC model...")
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_model, X_selected, y)

print("\nLinearSVC Cross-Validation Results:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1 Score: {svm_f1:.4f}")

# Evaluate KNeighborsClassifier
print("\nEvaluating KNeighborsClassifier model...")
knn_accuracy, knn_precision, knn_recall, knn_f1 = evaluate_model(knn_model, X_selected, y)

print("\nKNeighborsClassifier Cross-Validation Results:")
print(f"Accuracy: {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"F1 Score: {knn_f1:.4f}")

# Step 4: Final Model Training and Evaluation on Test Set
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train and predict with Naive Bayes
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# Train and predict with LinearSVC
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

# Train and predict with KNeighborsClassifier
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# Evaluate Naive Bayes on Test Set
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_conf_matrix = confusion_matrix(y_test, nb_pred)

print("\n\n\nNaive Bayes Model Evaluation on Test Set:")
print(f"Accuracy: {nb_accuracy:.4f}")
print(f"Precision: {nb_precision:.4f}")
print(f"Recall: {nb_recall:.4f}")
print(f"F1 Score: {nb_f1:.4f}")
print("Confusion Matrix:\n", nb_conf_matrix)

# Evaluate LinearSVC on Test Set
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_conf_matrix = confusion_matrix(y_test, svm_pred)

print("\nLinearSVC Model Evaluation on Test Set:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1 Score: {svm_f1:.4f}")
print("Confusion Matrix:\n", svm_conf_matrix)

# Evaluate KNeighborsClassifier on Test Set
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_conf_matrix = confusion_matrix(y_test, knn_pred)

print("\nKNeighborsClassifier Model Evaluation on Test Set:")
print(f"Accuracy: {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"F1 Score: {knn_f1:.4f}")
print("Confusion Matrix:\n", knn_conf_matrix)

# Visualization: Confusion Matrices
plt.figure(figsize=(18, 6))

# Naive Bayes Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')

# LinearSVC Confusion Matrix
plt.subplot(1, 3, 2)
sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.title("LinearSVC Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')

# KNeighborsClassifier Confusion Matrix 
plt.subplot(1, 3, 3)
sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Recommended', 'Recommended'], yticklabels=['Not Recommended', 'Recommended'])
plt.title("KNeighborsClassifier Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
