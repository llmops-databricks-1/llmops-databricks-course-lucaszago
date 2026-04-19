import mlflow
from mlflow.models import ModelConfig

from semantic_curator.agent import SemanticAgent

config = ModelConfig()

lakebase_project_id = None

agent = SemanticAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
    lakebase_project_id=lakebase_project_id,
)

mlflow.models.set_model(agent)
