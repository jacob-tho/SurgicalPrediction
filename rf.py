import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from Softwareprojekt import X_test, y_test, X_train, X_val, y_train, y_val, preprocessor
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE()),  # Platzhalter
    ('classifier', classifier)
])

'''
Hyperparametersuche
Unterschiedliche Resampling Strategien - SMOTE, SMOTEENN, ADASYN
Anzahl der Bäume und deren Struktur
Dokumentation der unterschiedlichen Hyperparameter: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
'''
param_grid = {
    'sampler': [None, SMOTE(random_state=42), ADASYN(random_state=42), SMOTEENN(random_state=42)],
    'classifier__n_estimators': [100, 300, 500],
    'classifier__criterion' : ['gini', 'entropy'],
    'classifier__max_features': ['sqrt', 'log2', None],
    'classifier__bootstrap': [True, False],
}

split_index = [-1] * len(X_train) + [0] * len(X_val)
ps = PredefinedSplit(test_fold=split_index)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring='f1',
    cv=ps,
    verbose=3,
    n_jobs=-1
)
grid_search.fit(X_combined, y_combined)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'random_forest_model.pkl')
