from prefect import flow
from prefect_pipeline import churn_pipeline_flow

if __name__ == "__main__":
    churn_pipeline_flow.deploy(
        name="churn-daily-deployment",
        work_pool_name="local-pool",
        cron="0 9 * * *",  # Tous les jours à 9h du matin
        parameters={"filepath": "Churn_Modelling.csv"}
    )
