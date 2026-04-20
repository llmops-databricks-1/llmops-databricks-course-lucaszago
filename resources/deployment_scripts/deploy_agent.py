# Databricks notebook source
from databricks import agents 
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils 

from loguru import logger 
from mlflow import MlflowClient

from semantic_curator.config import ProjectConfig

# COMMAND ----------
git_sha = dbutils.widgets.get("git_sha")
env = dbutils.widgets.get("env")
secret_scope = "semantic-agent-scope"

# Load configuration 
cfg = ProjectConfig.from_yaml("../../project_config.yml", env=env)

# Get more details 
model_name = f"{cfg.catalog}.{cfg.schema}.semantic_agent"
endpoint_name = f"semantic-agent-endpoint-{env}"

client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version


# Get experiment ID 
experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info("Deploying agent:")
logger.info(f" Model: {model_name}")
logger.info(f" Version: {model_version}")
logger.info(f" Endpoint: {endpoint_name}")

# COMMAND ----------
deploy_kwargs = {
    "model_name": model_name,
    "model_version": int(model_version),
    "endpoint_name": endpoint_name,
    "scale_to_zero": True,
    "workload_size": "Small",
    "deploy_feedback_model": False,
    "environment_vars": {
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{secrets/{secret_scope}/client_id}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{secrets/{secret_scope}/client_secret}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
}
if cfg.usage_policy_id:
    deploy_kwargs["usage_policy_id"] = cfg.usage_policy_id

agents.deploy(**deploy_kwargs)

logger.info(f"Deployment complete!")
