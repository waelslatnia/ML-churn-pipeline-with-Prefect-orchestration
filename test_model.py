import os
import pytest
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model


def test_prepare_data():
    """Vérifie que les données sont bien chargées et découpées."""
    assert os.path.exists("Churn_Modelling.csv"), "Fichier CSV manquant"
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_train_model():
    """Vérifie que le modèle s'entraîne correctement."""
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(X_train, y_train)
    assert model is not None


def test_evaluate_model():
    """Vérifie que l'accuracy est entre 0 et 1."""
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_save_and_load_model():
    """Vérifie la sauvegarde et le chargement du modèle."""
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(X_train, y_train)
    
    save_model(model, "test_churn_model.pkl")
    loaded = load_model("test_churn_model.pkl")
    
    assert loaded is not None
    os.remove("test_churn_model.pkl")
