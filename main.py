import joblib
import numpy as np
from Softwareprojekt import numeric_features, categorical_features, plot_heat, X_test, y_test
# Derzeit sehr viele falsch positive Werte

logistic_regression = joblib.load('logistic_regression_model.pkl')

# Feature Importance
importances = logistic_regression.named_steps['classifier'].coef_[0]
feature_names = numeric_features.tolist() + categorical_features.tolist()
importance_report = sorted(zip(importances, feature_names), reverse=True)

print("\nFeature Importances:")
for importance, feature in importance_report:
    print(f"{feature}: {importance:.4f}")

'''
OUTPUT
Feature Importances:
OPTIME: 0.3057
ANESTHES: 0.1852
SURGSPEC: 0.1154
WNDINF: 0.1107
DIABETES: 0.0841
INOUT: 0.0803
AGE: 0.0721
SMOKE: 0.0518
WTLOSS: 0.0489
DYSPNEA: 0.0339
TRANSFUS: 0.0307
STEROID: 0.0289
ASCITES: 0.0241
HXCOPD: 0.0226
SEX: 0.0201
RENAFAIL: 0.0184
HXCHF: 0.0108
BLEEDIS: 0.0075
Year: 0.0037
WEIGHT: 0.0000
FNSTATUS2: 0.0000
DIALYSIS: -0.0007
VENTILAT: -0.0072
BMI: -0.0094
PRSEPIS: -0.0111
HEIGHT: -0.0349
HYPERMED: -0.0487
RACE_NEW: -0.0858
DISCANCR: -0.8785
'''


xgb = joblib.load('XGB.pkl')
y_pred_test = xgb.predict(X_test)
plot_heat(y_test, y_pred_test)
# Feature Importance
importances = xgb.named_steps['classifier'].feature_importances_
feature_names = numeric_features.tolist() + categorical_features.tolist()
importance_report = sorted(zip(importances, feature_names), reverse=True)

print("\nFeature Importances:")
for importance, feature in importance_report:
    print(f"{feature}: {importance:.4f}")

'''
Feature Importances:
ANESTHES: 0.0288
Year: 0.0159
HEIGHT: 0.0112
SURGSPEC: 0.0105
OPTIME: 0.0099
PRSEPIS: 0.0078
FNSTATUS2: 0.0074
WTLOSS: 0.0043
ASCITES: 0.0041
INOUT: 0.0038
DYSPNEA: 0.0036
HYPERMED: 0.0036
HXCHF: 0.0036
WNDINF: 0.0035
RACE_NEW: 0.0033
DIABETES: 0.0033
SMOKE: 0.0031
SEX: 0.0031
HXCOPD: 0.0029
AGE: 0.0029
TRANSFUS: 0.0027
RENAFAIL: 0.0026
BLEEDIS: 0.0025
DIALYSIS: 0.0023
WEIGHT: 0.0023
VENTILAT: 0.0021
BMI: 0.0019
STEROID: 0.0016
DISCANCR: 0.0000
'''


random_forest = joblib.load('random_forest_model.pkl')
y_pred_test = random_forest.predict(X_test)
plot_heat(y_test, y_pred_test)
# Feature Importance
importances = xgb.named_steps['classifier'].feature_importances_
feature_names = numeric_features.tolist() + categorical_features.tolist()
importance_report = sorted(zip(importances, feature_names), reverse=True)

print("\nFeature Importances:")
for importance, feature in importance_report:
    print(f"{feature}: {importance:.4f}")

'''
ANESTHES: 0.0288
Year: 0.0159
HEIGHT: 0.0112
SURGSPEC: 0.0105
OPTIME: 0.0099
PRSEPIS: 0.0078
FNSTATUS2: 0.0074
WTLOSS: 0.0043
ASCITES: 0.0041
INOUT: 0.0038
DYSPNEA: 0.0036
HYPERMED: 0.0036
HXCHF: 0.0036
WNDINF: 0.0035
RACE_NEW: 0.0033
DIABETES: 0.0033
SMOKE: 0.0031
SEX: 0.0031
HXCOPD: 0.0029
AGE: 0.0029
TRANSFUS: 0.0027
RENAFAIL: 0.0026
BLEEDIS: 0.0025
DIALYSIS: 0.0023
WEIGHT: 0.0023
VENTILAT: 0.0021
BMI: 0.0019
STEROID: 0.0016
DISCANCR: 0.0000
'''
