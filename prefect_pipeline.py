from prefect import task, flow
from model_pipeline import prepare_data, train_model, evaluate_model, save_model


@task(name="prepare-data-task")
def task_prepare_data(filepath):
    """Charge et prepare les donnees depuis le fichier CSV."""
    return prepare_data(filepath)


@task(name="train-model-task")
def task_train_model(X_train, y_train):
    """Entraine le modele Random Forest."""
    return train_model(X_train, y_train)


@task(name="evaluate-model-task")
def task_evaluate_model(model, X_test, y_test):
    """Evalue le modele et retourne l accuracy."""
    return evaluate_model(model, X_test, y_test)


@task(name="save-model-task")
def task_save_model(model):
    """Sauvegarde le modele entraine."""
    save_model(model)


@flow(name="ml-churn-pipeline-flow", log_prints=True)
def churn_pipeline_flow(filepath="Churn_Modelling.csv"):
    """Flow principal orchestrant le pipeline ML de prediction de churn."""
    print("Lancement du pipeline Churn via Prefect")

    X_train, X_test, y_train, y_test = task_prepare_data(filepath)
    model = task_train_model(X_train, y_train)
    accuracy = task_evaluate_model(model, X_test, y_test)
    task_save_model(model)

    print(f"Pipeline termine. Accuracy finale : {accuracy:.4f}")
    return {"accuracy": float(accuracy), "status": "success"}


if __name__ == "__main__":
    churn_pipeline_flow()
