"""
Module d'intégration MLflow -> Elasticsearch.
Envoie les métriques, paramètres et métadonnées d'un run MLflow vers un index Elasticsearch
pour visualisation ultérieure dans Kibana.
"""
import datetime
from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:9200"
ES_INDEX = "mlflow-metrics"


def get_es_client():
    """Retourne un client Elasticsearch connecté."""
    return Elasticsearch(ES_HOST)


def ensure_index_exists(es_client, index_name: str = ES_INDEX):
    """Crée l'index avec son mapping si celui-ci n'existe pas encore."""
    if es_client.indices.exists(index=index_name):
        return
    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
    mappings = {
        "properties": {
            "@timestamp": {"type": "date"},
            "run_id": {"type": "keyword"},
            "run_name": {"type": "keyword"},
            "experiment_id": {"type": "keyword"},
            "experiment_name": {"type": "keyword"},
            "status": {"type": "keyword"},
            "duration_seconds": {"type": "float"},
            "params": {"type": "object"},
            "metrics": {"type": "object"},
        }
    }
    es_client.indices.create(index=index_name, settings=settings, mappings=mappings)


def log_run_to_elasticsearch(run_id: str, run_name: str, experiment_id: str,
                              experiment_name: str, params: dict, metrics: dict,
                              status: str = "FINISHED", duration_seconds: float = None):
    """
    Envoie les informations d'un run MLflow vers Elasticsearch.

    Args:
        run_id (str): ID du run MLflow.
        run_name (str): Nom du run.
        experiment_id (str): ID de l'expérience MLflow.
        experiment_name (str): Nom de l'expérience MLflow.
        params (dict): Hyperparamètres loggés (ex: {"n_estimators": 100}).
        metrics (dict): Métriques loggées (ex: {"accuracy": 0.86}).
        status (str): Statut du run (FINISHED, FAILED, etc.).
        duration_seconds (float): Durée d'exécution du run en secondes.
    """
    es = get_es_client()
    ensure_index_exists(es)

    document = {
        "@timestamp": datetime.datetime.utcnow().isoformat(),
        "run_id": run_id,
        "run_name": run_name,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "status": status,
        "duration_seconds": duration_seconds,
        "params": {str(k): str(v) for k, v in params.items()},
        "metrics": {str(k): float(v) for k, v in metrics.items()},
    }

    # Ajoute aussi chaque métrique/paramètre à plat pour faciliter les filtres Kibana
    for k, v in metrics.items():
        document[f"metric_{k}"] = float(v)
    for k, v in params.items():
        document[f"param_{k}"] = str(v)

    es.index(index=ES_INDEX, document=document)
    print(f"📊 Run '{run_name}' ({run_id}) envoyé vers Elasticsearch (index: {ES_INDEX})")
