import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from Softwareprojekt import X,y

X = X
y = y
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)

preprocessor = ColumnTransformer(
            transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop="first", sparse_output=False), categorical_features)#Oder binary-Encoding?
            ]
)

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

smote = SMOTE(random_state=42)
sgd_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('classifier', SGDClassifier(random_state=42))
])

param_grid = {
    'classifier__loss': ['hinge', 'log', 'perceptron', 'squared_hinge'],  # Verlustfunktion
    'classifier__penalty': ['l2', 'l1', 'elasticnet'],  # Regularisierung
    'classifier__alpha': [1e-5, 1e-4, 1e-3, 1e-2],  # Regulsarisierung
    'classifier__learning_rate': ['constant', 'optimal', 'invscaling'],  # Lernratenänderung
    'classifier__eta0': [0.001, 0.01, 0.1],  # Anfangs-Lernrate
    'classifier__max_iter': [2000],
    'classifier__class_weight': ['balanced']
}


grid_search = GridSearchCV(sgd_pipeline,
                            param_grid,
                            cv=5,
                            scoring='f1',
                            n_jobs=-3,
                            verbose=3
)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters from the grid search
print(f"Best Hyperparameters: {grid_search.best_params_}")


best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report: ")
print(classification_report(y_test, y_pred))
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

joblib.dump(best_model, 'SGDClassifier_model.pkl')

importances = best_model.named_steps['classifier'].coef_[0]
feature_names = numeric_features.tolist() + categorical_features.tolist()
importance_report = sorted(zip(importances, feature_names), reverse=True)

print("\nFeature Importances:")
for importance, feature in importance_report:
    print(f"{feature}: {importance:.4f}")

'''
Best Hyperparameters: {'classifier__alpha': 0.01, 'classifier__class_weight': 'balanced', 'classifier__eta0': 0.001,
 'classifier__learning_rate': 'invscaling', 'classifier__loss': 'hinge', 'classifier__max_iter': 2000,
 'classifier__penalty': 'l1'}

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.70      0.81     35190
           1       0.08      0.61      0.13      1421

    accuracy                           0.69     36611
   macro avg       0.53      0.65      0.47     36611
weighted avg       0.94      0.69      0.79     36611

Test Accuracy: 0.6933
Feature Importances:
OPTIME: 0.4899, WNDINF: 0.1232, Year: -0.0574, WEIGHT: -0.0669
'''
