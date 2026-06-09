from prefect import task, flow
from model_pipeline import prepare_data, train_model, evaluate_model, save_model

@task(name="prepare-data-task")
def task_prepare_data(filepath):
    return prepare_data(filepath)

@task(name="train-model-task")
def task_train_model(X_train, y_train):
    return train_model(X_train, y_train)

@task(name="evaluate-model-task")
def task_evaluate_model(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)

@task(name="save-model-task")
def task_save_model(model):
    save_model(model)

@flow(name="all", log_prints=True)
def pipeline_full(filepath="Churn_Modelling.csv"):
    X_train, X_test, y_train, y_test = task_prepare_data(filepath)
    model = task_train_model(X_train, y_train)
    accuracy = task_evaluate_model(model, X_test, y_test)
    task_save_model(model)
    print(f"Full pipeline finished. Final accuracy: {accuracy:.4f}")
    return {"accuracy": float(accuracy), "status": "success"}

@flow(name="Entrainement", log_prints=True)
def entrainement_flow(filepath="Churn_Modelling.csv"):
    X_train, X_test, y_train, y_test = task_prepare_data(filepath)
    model = task_train_model(X_train, y_train)
    print("Training completed")
    return model

@flow(name="Evaluate", log_prints=True)
def evaluate_flow(filepath="Churn_Modelling.csv"):
    # Load existing data
    X_train, X_test, y_train, y_test = task_prepare_data(filepath)
    # Assuming model is loaded from file or generated - here, we retrain for demo
    model = task_train_model(X_train, y_train)
    accuracy = task_evaluate_model(model, X_test, y_test)
    print(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    import sys
    flow_type = sys.argv[1] if len(sys.argv) > 1 else "full"
    if flow_type == "full":
        pipeline_full()
    elif flow_type == "entrainement":
        entrainement_flow()
    elif flow_type == "evaluate":
        evaluate_flow()
    else:
        print("Unknown flow type. Use 'full', 'entrainement', or 'evaluate'.")
