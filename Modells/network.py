import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import optuna
import numpy as np
import pandas as pd
from Softwareprojekt import X,y


# Seeds für Reproduzierbarkeit
torch.manual_seed(42)
np.random.seed(42)

'''
# Noch einmal, da sich PyTorch anders verhält als sklearn und Preprocessor Datentyp nicht annimmt
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_val_cat = encoder.transform(X_val[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_val_num = scaler.transform(X_val[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

X_train = np.hstack([X_train_num, X_train_cat])
X_val = np.hstack([X_val_num, X_val_cat])
X_test = np.hstack([X_test_num, X_test_cat])

# Unterschiedliche Aktivierungsfunktionen für Hyperparameter-Training
# Sind andere besser? -- Swish z.B.?
ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU(),
    "SiLU": nn.SiLU(),
    "Sigmoid": nn.Sigmoid()
}

# Grundlegendes Modell / Architektur
# Layeranzahl und -größe, Normalisierungslayer, Aktivierungsfuntion und Regularisierung
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_fns, dropout_rate):
        super(BinaryClassifier, self).__init__()
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(ACTIVATION_FUNCTIONS[activation_fns[i]])
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Optuna für Hpyerparameter-Optimierung.
# Dokumentation auf https://optuna.org/ einsehbar
def main():
    def objective(trial):

        sampler = SMOTE(random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
        y_tensor = torch.tensor(y_resampled.values, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Unterschiedliche Hpyerparameter:
        # Anzahl der Layer, Größe der Layer, Aktivierungsfunktionen, Optimierungsverfahren
        # Verlustfunktion, Lernrate, Regularisierungsrate
        num_layers = trial.suggest_int("num_layers", 2, 6)
        hidden_sizes = [trial.suggest_categorical(f"hidden_size_{i}", [64, 128, 256, 512, 1024]) for i in range(num_layers)]
        activation_fns = [trial.suggest_categorical(f"activation_{i}", list(ACTIVATION_FUNCTIONS.keys())) for i in range(num_layers)]
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"])
        loss_function_name = trial.suggest_categorical("loss_function", ["BCEWithLogitsLoss", "HingeEmbeddingLoss"]) #Binary Cross Entropy und Hinge Loss
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

        model = BinaryClassifier(input_size=X_train.shape[1], hidden_sizes=hidden_sizes, activation_fns=activation_fns, dropout_rate=dropout_rate)
        loss_fn = getattr(nn, loss_function_name)()
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(10):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor)
            y_scores = torch.sigmoid(outputs).squeeze().numpy()
            precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
            # Leicht abgeänderter Threshold (standardmäßig bei 0.5), da bei unbalancierten Datensätzen Klasse 0 einen Bias hat
            best_threshold = thresholds[np.argmax(2 * (precision * recall) / (precision + recall + 1e-8))]
            y_pred_binary = (y_scores > best_threshold).astype(int)

        return f1_score(y_val, y_pred_binary, pos_label=1)

    # Hyperparametersuche
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Das "beste" Modell wird ab hier nochmal vollständig trainiert
    best_params = study.best_params
    torch.save(best_params, "best_model_params.pth")
    hidden_sizes = [best_params[f"hidden_size_{i}"] for i in range(best_params["num_layers"])]
    activation_fns = [best_params[f"activation_{i}"] for i in range(best_params["num_layers"])]

    final_model = BinaryClassifier(input_size=X_train.shape[1],
                                   hidden_sizes=hidden_sizes,
                                   activation_fns=activation_fns,
                                   dropout_rate=best_params["dropout_rate"])

    loss_fn = getattr(nn, best_params["loss_function"])()
    optimizer = getattr(optim, best_params["optimizer"])(final_model.parameters(), lr=best_params["learning_rate"])

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    final_model.train()
    for epoch in range(30):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
    torch.save(final_model.state_dict(), "best_neural_net.pth")

if __name__ == '__main__':
    main()
