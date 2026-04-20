# Databricks notebook source
from mlflow import MlflowClient
import os
import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient, dbutils
from loguru import logger
from mlflow import MlflowClient

from semantic_curator.config import ProjectConfig


# Setup MLFLOW TRACKING 
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv
    load_dotenv()
    profile = os.getenv("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

cfg = ProjectConfig.from_yaml("../project_config.yml")
env = dbutils.widgets.get("env")
model_name = f"{cfg.catalog}.{cfg.schema}.semantic_agent"
endpoint_name = f"semantic-agent-endpoint-{env}-course"
secret_scope = "semantic-agent-scope"

model_version = MlflowClient().get_model_version_by_alias(
    model_name, "latest-model"
).version
workspace = WorkspaceClient()
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_name)

# COMMAND ----------
git_sha = dbutils.widgets.get("git_sha")

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

agents.deploy(
    **deploy_kwargs,
)

# COMMAND ----------
import random 
from datetime import datetime
from openai import OpenAI

host = workspace.config.host 
token = workspace.tokens.create(lifetime_seconds=2000).token_value

client = OpenAI(api_key=token, 
                base_url=f"{host}/serving-endpoints",
                )

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content":"What are recent papers about LLMs and reasoning?"}
    ], 
    extra_body={"custom_inputs":{
        "session_id": session_id,
        "request_id": request_id
    }}
    )

logger.info(f"Response ID: {response.id}")
logger.info(f"Session ID: {response.custom_outputs.get('session_id')}")
logger.info(f"Request ID: {response.custom_outputs.get('request_id')}")
logger.info("\nAssistant Response:")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)

# COMMAND ----------
