import os

import mlflow
from mlflow.models import ModelConfig

from semantic_curator.agent import SemanticAgent

config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "lukaszag",
        "genie_space_id": None,
        "system_prompt": (
            "You are a helpful AI assistant that helps users find "
            "and understand research papers."
        ),
        "llm_endpoint": "databricks-llama-4-maverick",
        "lakebase_project_id": "semantic-agent-lakebase-dev",
    }
)

lakebase_project_id = None
if all(
    [
        os.environ.get("LAKEBASE_SP_CLIENT_ID"),
        os.environ.get("LAKEBASE_SP_CLIENT_SECRET"),
        os.environ.get("LAKEBASE_SP_HOST"),
        os.environ.get("MODEL_SERVING_ENDPOINT_NAME"),
    ]
):
    lakebase_project_id = config.get("lakebase_project_id")

agent = SemanticAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
    lakebase_project_id=lakebase_project_id,
)
mlflow.models.set_model(agent)
