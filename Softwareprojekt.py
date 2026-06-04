import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

### Preprocessing

def plot_heat(y_true, y_pred):
    '''
    Gibt eine Visualisierung und Veranschaulichung der Performance eines Modells an.
    Da y_true ein Funktionsargument ist, funktioniert das offensichtlich nur für überwachte Lernverfahren.
    '''

    cm = confusion_matrix(y_true, y_pred)
    print(f"Genauigkeit: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F-1 Score für 1-Class: {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Keine Komplikation', 'Komplikation'])
    plt.xlabel('Vorhersage')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.show()

df = pd.read_csv("ACS-NSQIP_synthetic_large.csv")
pd.set_option('display.max_columns', None)
for col in df.columns:
    if set(df[col].unique()) == {'Yes', 'No'}:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
#print(df.info()) #AGE Column ist Typ Object
#df1 = df[df.isna().any(axis=1)]
# Age column hat "viele" NaN-Werte (etwa 450)
#del df1
df = df.dropna()

#df1 = df.loc[df["AGE"] == "90+"]
# Etwa 2900 Werte -> Ändern oder verzichten?
#del df1
df = df[df["AGE"] != "90+"]
df["AGE"] = df["AGE"].astype(int)
df.DISCANCR = df.DISCANCR.replace(to_replace="1", value="Yes")

#Betrachte Verteilung der Kategorischen Daten und Verteilung der letzten Spalte
#Relevant für spätere graphische Betrachtungen
'''
race = df["RACE_NEW"].value_counts()
status = df["FNSTATUS2"].value_counts()
discan = df["DISCANCR"].value_counts()
sepsis = df["PRSEPIS"].value_counts()
anesthesis = df["ANESTHES"].value_counts()
surgery = df["SURGSPEC"].value_counts()
inout = df["INOUT"].value_counts()
year = df["Year"].value_counts()
sex = df["SEX"].value_counts()
complication = df["surgical1"].value_counts()
'''


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(f"Anteil in 0-Class: {len(y.loc[y==0].index.tolist())/y.shape[0]*100}%")
print(f"Anteil in 1-Class: {len(y.loc[y==1].index.tolist())/y.shape[0]*100}%")
print("Sehr ungleich verteilter Datensatz!")

# Aufteilen in Trainings, Validierungs und Test Datensatz
# Training und Validierung für Hyperparametersuche
# Trainingsdaten für Training des finalen Modells
# Testdaten werden erst zum Schluss verwendet, um unabhängig vom Training zu sein
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop="first", sparse_output=False), categorical_features)
    ]
)
