import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVC
from Softwareprojekt import X_test, y_test, X_train, X_val, y_train, y_val, preprocessor, plot_heat
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
# Nicht mit eingebracht, da Laufzeit zu hoch

X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

classifier = SVC(random_state=42, probability=True)

pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE()),  # Platzhalter
    ('classifier', classifier)
])


param_grid = {
    'sampler': [SMOTE(random_state=42), ADASYN(random_state=42), SMOTEENN(random_state=42), RandomUnderSampler(random_state=42)],
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__kernel': ['rbf', 'sigmoid'],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # For 'rbf', 'poly', 'sigmoid'
    'classifier__coef0': [0.0, 0.1, 0.5, 1.0],  # Only for 'poly' and 'sigmoid' kernels
}

split_index = [-1] * len(X_train) + [0] * len(X_val)  # -1 for training, 0 for validation
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

# Best model and evaluation
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

#y_pred_val = best_model.predict(X_val)
y_pred_test = best_model.predict(X_test)

joblib.dump(best_model, 'svm_model.pkl')
