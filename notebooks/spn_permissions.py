# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel

from semantic_curator.config import ProjectConfig
from semantic_curator.utils.common import get_widget

env = get_widget("env", "dev")
spn_secret_scope = get_widget("spn_secret_scope", "semantic-agent-scope")
semantic_agent_scope = spn_secret_scope

cfg = ProjectConfig.from_yaml("../project_config.yml", env=env)
w = WorkspaceClient()

# COMMAND ----------
spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")

# COMMAND ----------
vs_endpoint = w.vector_search_endpoints.get_endpoint(cfg.vector_search_endpoint)
w.permissions.update(
    request_object_type="vector-search-endpoints",
    request_object_id=vs_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

# COMMAND ----------
w.permissions.update(
    request_object_type="warehouses",
    request_object_id=cfg.warehouse_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)
