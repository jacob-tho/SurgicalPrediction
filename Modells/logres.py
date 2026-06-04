import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from Softwareprojekt import X_test, y_test, X_train, X_val, y_train, y_val, preprocessor



X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

# Baseline Algorithmus: Logistische Regression
classifier = LogisticRegression(random_state=42, max_iter=1000)
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE()), #Platzhalter
    ('classifier', classifier)
])

'''
Hyperparametersuche
Unterschiedliche Resampling Strategien - SMOTE, SMOTEENN, ADASYN
Mehrere Regularisierungsraten und Regularisierungsverfahren
Optimierungsverfahren
Dokumentation der unterschiedlichen Hyperparameter: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
'''
param_grid = {
    'sampler': [SMOTE(random_state=42), ADASYN(random_state=42), SMOTEENN(random_state=42)],#Unterschiedliche Resampling Techniken
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': [None, 'l1', 'l2'],
    'classifier__solver': ['liblinear', 'newton-cholesky', 'saga'],
    'classifier__class_weight': ['balanced', {0:1, 1:10}, {0:1, 1:15}]
}

split_index = [-1] * len(X_train) + [0] * len(X_val) #Training und Validierung unabhängig von Testdatensatz
ps = PredefinedSplit(test_fold=split_index)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1', # Nach f1-Score optimieren
    cv=ps,
    verbose=1,
    n_jobs=-2
)
grid_search.fit(X_combined, y_combined)

best_model = grid_search.best_estimator_
#Speichern des Modells
joblib.dump(best_model, 'logistic_regression_model.pkl')
