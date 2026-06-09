from prefect import serve
from prefect_pipeline import pipeline_full # Changed from churn_pipeline_flow

if __name__ == "__main__":
    deployment = pipeline_full.to_deployment( # Changed to pipeline_full
        name="churn-pipeline-deployment",
        cron="0 9 * * *",
    )
    serve(deployment)
