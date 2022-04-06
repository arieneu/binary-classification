import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics


# Read our training and test data 
training_data = pd.read_csv(str(sys.argv[1]), sep='\t')
test_data = pd.read_csv(str(sys.argv[2]), sep='\t')

# Grab our attributes and outputs, i.e. X and y
X = training_data.drop('Class', axis=1)
y = training_data['Class']

# Scale our data
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Define a scoring function for our models
def get_score(model, X_train, X_test, y_train, y_test):
  model.fit(X_train, y_train)
  return model.score(X_test, y_test)

# Create our Logistic Regression model
log_model = LogisticRegression()

# Create our SVC model
svc_model = SVC(probability=True)
param_grid_svc = [{
    'C' : [1, 1.1, 1.2]
}]
grid_search_svc = GridSearchCV(svc_model, param_grid_svc, scoring='accuracy', cv=3)
grid_search_svc.fit(X_scaled, y)
final_svc = grid_search_svc.best_estimator_
print(f'Optimal hyperparameters for SVC: {grid_search_svc.best_params_}')

# Create our Ridge model
ridge_model = RidgeClassifier()
param_grid_ridge = [{
    'alpha': [1, 5, 10, 20, 30, 50, 100]
}]
grid_search_ridge = GridSearchCV(ridge_model, param_grid_ridge, scoring='accuracy', cv=3)
grid_search_ridge.fit(X_scaled, y)
final_ridge = grid_search_ridge.best_estimator_
print(f'Optimal hyperparameters for Ridge Classification: {grid_search_ridge.best_params_}')

# Create an 10-Fold Stratified Cross-Validation object
skf = StratifiedKFold(n_splits=10)

# Create variables to hold our performance metrics
log_tprs = []
log_precisions = []
log_accuracies = []
log_aucrocs = []
log_aucprs = []
mean_fpr = np.linspace(0, 1, 100)

svc_tprs = []
svc_precisions = []
svc_accuracies = []
svc_aucrocs = []
svc_aucprs = []

ridge_tprs = []
ridge_precisions = []
ridge_accuracies = []
ridge_aucrocs = []
ridge_aucprs = []

# 10-Fold Stratified Cross-Validation
for train_index, test_index in skf.split(X_scaled, y):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  #Logistic Regression
  log_score = get_score(log_model, X_train, X_test, y_train, y_test)
  log_accuracies.append(log_score)
  log_dec = log_model.decision_function(X_test)
  
  log_fpr, log_tpr, _ = metrics.roc_curve(y_test, log_dec)
  interp_log_tpr = np.interp(mean_fpr, log_fpr, log_tpr)
  interp_log_tpr[0] = 0.0
  log_tprs.append(interp_log_tpr)
  log_aucrocs.append(metrics.roc_auc_score(y_test, log_dec))

  log_precision, log_recall, _ = metrics.precision_recall_curve(y_test, log_dec)
  interp_log_precision = np.interp(mean_fpr, log_precision, log_recall)
  interp_log_precision[0] = 1
  log_precisions.append(interp_log_precision)
  log_aucprs.append(metrics.auc(log_recall, log_precision))

  #SVC
  svc_score = get_score(final_svc, X_train, X_test, y_train, y_test)
  svc_accuracies.append(svc_score)
  svc_dec = final_svc.decision_function(X_test)
  
  svc_fpr, svc_tpr, _ = metrics.roc_curve(y_test, svc_dec)
  interp_svc_tpr = np.interp(mean_fpr, svc_fpr, svc_tpr)
  interp_svc_tpr[0] = 0.0
  svc_tprs.append(interp_svc_tpr)
  svc_aucrocs.append(metrics.roc_auc_score(y_test, svc_dec))

  svc_precision, svc_recall, _ = metrics.precision_recall_curve(y_test, svc_dec)
  interp_svc_precision = np.interp(mean_fpr, svc_precision, svc_recall)
  interp_svc_precision[0] = 1
  svc_precisions.append(interp_svc_precision)
  svc_aucprs.append(metrics.auc(svc_recall, svc_precision))

  #Ridge
  ridge_score = get_score(final_ridge, X_train, X_test, y_train, y_test)
  ridge_accuracies.append(ridge_score)
  ridge_dec = final_ridge.decision_function(X_test)
  
  ridge_fpr, ridge_tpr, _ = metrics.roc_curve(y_test, ridge_dec)
  interp_ridge_tpr = np.interp(mean_fpr, ridge_fpr, ridge_tpr)
  interp_ridge_tpr[0] = 0.0
  ridge_tprs.append(interp_ridge_tpr)
  ridge_aucrocs.append(metrics.roc_auc_score(y_test, ridge_dec))

  ridge_precision, ridge_recall, _ = metrics.precision_recall_curve(y_test, ridge_dec)
  interp_ridge_precision = np.interp(mean_fpr, ridge_precision, ridge_recall)
  interp_ridge_precision[0] = 1
  ridge_precisions.append(interp_ridge_precision)
  ridge_aucprs.append(metrics.auc(ridge_recall, ridge_precision))  
  
# Compute the means for our performance metrics

# Logistic Regression
mean_log_tpr = np.mean(log_tprs, axis=0)
mean_log_tpr[-1] = 1.0

mean_log_aucroc = metrics.auc(mean_fpr, mean_log_tpr)
std_log_aucroc = np.std(log_aucrocs)

mean_log_precision = np.mean(log_precisions, axis=0)
mean_log_precision[-1] = 0.0

mean_log_aucpr = metrics.auc(mean_log_tpr, mean_log_precision)
std_log_aucpr = np.std(log_aucprs)

mean_log_accuracy = np.mean(log_accuracies)
std_log_accuracy = np.std(log_accuracies)

# SVC
mean_svc_tpr = np.mean(svc_tprs, axis=0)
mean_svc_tpr[-1] = 1.0

mean_svc_aucroc = metrics.auc(mean_fpr, mean_svc_tpr)
std_svc_aucroc = np.std(svc_aucrocs)

mean_svc_precision = np.mean(svc_precisions, axis=0)
mean_svc_precision[-1] = 0.0

mean_svc_aucpr = metrics.auc(mean_svc_tpr, mean_svc_precision)
std_svc_aucpr = np.std(svc_aucprs)

mean_svc_accuracy = np.mean(svc_accuracies)
std_svc_accuracy = np.std(svc_accuracies)

# Ridge
mean_ridge_tpr = np.mean(ridge_tprs, axis=0)
mean_ridge_tpr[-1] = 1.0

mean_ridge_aucroc = metrics.auc(mean_fpr, mean_ridge_tpr)
std_ridge_aucroc = np.std(ridge_aucrocs)

mean_ridge_precision = np.mean(ridge_precisions, axis=0)
mean_ridge_precision[-1] = 0.0

mean_ridge_aucpr = metrics.auc(mean_ridge_tpr, mean_ridge_precision)
std_ridge_aucpr = np.std(ridge_aucprs)

mean_ridge_accuracy = np.mean(ridge_accuracies)
std_ridge_accuracy = np.std(ridge_accuracies)

# Plot the ROC and PR curves

# Logistic Regression
print(f'Mean logisitic regression accuracy: {mean_log_accuracy} \u00B1 {std_log_accuracy}', end='\n\n')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
ax1.plot(mean_fpr, mean_log_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_log_aucroc, std_log_aucroc), lw=2, alpha=0.8)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.set_title("ROC (Logisitic Regression)")
ax1.legend()

ax2.plot(mean_fpr, mean_log_precision, color='b', label='Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_log_aucpr, std_log_aucpr), lw=2, alpha=0.8)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('PR (Logistic Regression)')
ax2.legend()

plt.show()

# SVC
print(f'Mean SVC accuracy: {mean_svc_accuracy} \u00B1 {std_svc_accuracy}', end='\n\n')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
ax1.plot(mean_fpr, mean_svc_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_svc_aucroc, std_svc_aucroc), lw=2, alpha=0.8)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.set_title("ROC (SVC)")
ax1.legend()

ax2.plot(mean_fpr, mean_svc_precision, color='b', label='Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_svc_aucpr, std_svc_aucpr), lw=2, alpha=0.8)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('PR (SVC)')
ax2.legend()

plt.show()

# Ridge
print(f'Mean Ridge accuracy: {mean_ridge_accuracy} \u00B1 {std_ridge_accuracy}', end='\n\n')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
ax1.plot(mean_fpr, mean_ridge_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_ridge_aucroc, std_ridge_aucroc), lw=2, alpha=0.8)
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.set_title("ROC (Ridge)")
ax1.legend()

ax2.plot(mean_fpr, mean_ridge_precision, color='b', label='Mean PR (AUC = %0.2f $\pm$ %0.2f)' % (mean_ridge_aucpr, std_ridge_aucpr), lw=2, alpha=0.8)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('PR (Ridge)')
ax2.legend()

plt.show()

# Compute the pobablility of being in class 1 using the test_data with out final model
probabilites = final_svc.predict_proba(test_data)
final_proba = probabilites[:, 1]