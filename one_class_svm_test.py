import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer
from Softwareprojekt import X_test, y_test, X_train, X_val, y_train, y_val, preprocessor, plot_heat

# Combine training and validation data for unsupervised learning
X_combined = pd.concat([X_train, X_val])
y_combined = np.concatenate([y_train, y_val])

# Define One-Class SVM classifier
classifier = OneClassSVM()

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# Hyperparameter grid for One-Class SVM tuning
param_grid = {
    'classifier__kernel': ['rbf', 'sigmoid'],
    'classifier__nu': [0.04],  # Fraction of outliers
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient
}

# PredefinedSplit for consistent validation split
split_index = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold=split_index)

# ROC-AUC scorer
def auc_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return roc_auc_score(y, y_pred)

scorer = make_scorer(lambda est, X: auc_scorer(est, X, y_combined), greater_is_better=True)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring=scorer,
    cv=ps,
    verbose=3,
    n_jobs=-1
)
grid_search.fit(X_combined, y_combined)

# Best model and evaluation
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict outliers (-1) and inliers (1) on test data
y_pred_test = best_model.predict(X_test)
roc_auc_test = roc_auc_score(y_test, y_pred_test)
print("Test ROC-AUC Score:", roc_auc_test)

# Save the best model
joblib.dump(best_model, 'one_class_svm_model.pkl')
