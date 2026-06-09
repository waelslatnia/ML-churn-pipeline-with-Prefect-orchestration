import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(filepath):
    """
    Charge, encode et standardise les donnees depuis un fichier CSV.
    
    Args:
        filepath (str): Chemin vers le fichier CSV.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")
    if not filepath.endswith('.csv'):
        raise ValueError("Le fichier doit etre au format CSV.")
    
    df = pd.read_csv(filepath)
    
    if "Exited" not in df.columns:
        raise ValueError("La colonne 'Exited' est absente du dataset.")
    
    encoder = LabelEncoder()
    if "Gender" in df.columns:
        df["Gender"] = encoder.fit_transform(df["Gender"])
    
    cols_to_drop = ["RowNumber", "CustomerId", "Surname", "Geography"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    joblib.dump(encoder, "gender_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Entraine un modele RandomForestClassifier.
    
    Args:
        X_train (array-like): Donnees d'entrainement standardisees.
        y_train (array-like): Cibles d'entrainement.
        
    Returns:
        RandomForestClassifier: Le modele entraine.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalue les performances du modele sur le jeu de test.
    
    Args:
        model: Modele entraine.
        X_test (array-like): Donnees de test standardisees.
        y_test (array-like): Cibles de test.
        
    Returns:
        float: Valeur de l'accuracy.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Accuracy : {accuracy:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions))
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, predictions))
    
    return accuracy


def save_model(model, filename="churn_model.pkl"):
    """
    Sauvegarde le modele entraine sur le disque avec joblib.
    
    Args:
        model: Modele entraine a sauvegarder.
        filename (str): Nom du fichier de sortie.
    """
    joblib.dump(model, filename)
    print(f"Modele sauvegarde dans {filename}")


def load_model(filename="churn_model.pkl"):
    """
    Charge un modele prealablement sauvegarde depuis le disque.
    
    Args:
        filename (str): Chemin vers le fichier du modele.
        
    Returns:
        object: Le modele charge.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Le fichier modele {filename} est introuvable.")
    
    model = joblib.load(filename)
    print("Modele charge avec succes")
    return model
