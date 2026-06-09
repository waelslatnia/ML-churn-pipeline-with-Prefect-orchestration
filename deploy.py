from prefect import serve
from prefect_pipeline import churn_pipeline_flow
if __name__ == "__main__":
    deployment = churn_pipeline_flow.to_deployment(
        name="churn-pipeline-deployment",
        cron="0 9 * * *",
    )
    serve(deployment)
