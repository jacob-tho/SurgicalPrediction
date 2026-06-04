import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from Softwareprojekt import X_test, y_test, X_train, X_val, y_train, y_val, preprocessor
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])


classifier = XGBClassifier(
    random_state=42,
    eval_metric='logloss' # Wäre hier eine andere Metrik angebrachter?
)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE()),  # Platzhalter
    ('classifier', classifier)
])

'''
Hyperparametersuche
Unterschiedliche Resampling Strategien - SMOTE, SMOTEENN, ADASYN
Mehrere Kombination für Baumstruktur (Anzahl der Knoten und Fan-Out)
Lernraten
Dokumentation der unterschiedlichen Hyperparameter: https://xgboost.readthedocs.io/en/stable/get_started.html
'''
param_grid = {
    'sampler': [SMOTE(random_state=42), ADASYN(random_state=42), SMOTEENN(random_state=42)],
    'classifier__max_depth': [10],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0.1, 0.3, 0.5],
    #'classifier__reg_alpha': [0.01, 0.1, 1],
    #'classifier__reg_lambda': [1, 1.5, 2],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__subsample': [0.6, 0.8, 1],
    'classifier__colsample_bytree': [0.6, 0.8, 1]
}
# Regularisierung auskommentiert, da es auch ohne über 9 Stunden gedauert hat


split_index = [-1] * len(X_train) + [0] * len(X_val)  # -1 für Training, 0 für Validierung
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
joblib.dump(best_model, 'XGB.pkl')
