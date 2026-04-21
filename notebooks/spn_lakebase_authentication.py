# Databricks notebook source
import urllib.parse

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
    Role,
    RoleAuthMethod,
    RoleIdentityType,
    RoleRoleSpec,
)
from google.protobuf.duration_pb2 import Duration
from loguru import logger

from semantic_curator.config import ProjectConfig
from semantic_curator.utils.common import get_widget

env = get_widget("env", "dev")
spn_secret_scope = get_widget("spn_secret_scope", "dev_SPN")

cfg = ProjectConfig.from_yaml("../project_config.yml", env=env)
w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id
client_id = dbutils.secrets.get(spn_secret_scope, "client_id")

# COMMAND ----------
try:
    project = pg_api.get_project(name=f"projects/{project_id}")
except Exception:
    project_spec = ProjectSpec(
        display_name=project_id,
        default_endpoint_settings=ProjectDefaultEndpointSettings(
            autoscaling_limit_min_cu=1,
            autoscaling_limit_max_cu=4,
            suspend_timeout_duration=Duration(seconds=300),
        ),
    )
    if cfg.usage_policy_id:
        project_spec.budget_policy_id = cfg.usage_policy_id

    project = pg_api.create_project(
        project_id=project_id,
        project=Project(spec=project_spec),
    ).wait()

default_branch = next(iter(pg_api.list_branches(parent=project.name)))
branch_parent = default_branch.name

try:
    pg_api.create_role(
        parent=branch_parent,
        role=Role(
            spec=RoleRoleSpec(
                identity_type=RoleIdentityType.SERVICE_PRINCIPAL,
                auth_method=RoleAuthMethod.LAKEBASE_OAUTH_V1,
                postgres_role=client_id,
            )
        ),
        role_id="semantic-agent-spn",
    ).wait()
    logger.info("Created Lakebase SPN role semantic-agent-spn.")
except Exception as exc:
    logger.info(f"Lakebase SPN role may already exist: {exc}")

# COMMAND ----------
endpoint = next(iter(pg_api.list_endpoints(parent=branch_parent)))
host = endpoint.status.hosts.host
pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)

user = w.current_user.me()
username = urllib.parse.quote_plus(user.user_name)

conn_string = (
    f"postgresql://{username}:{pg_credential.token}@{host}:5432/"
    "databricks_postgres?sslmode=require"
)

with psycopg.connect(conn_string) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages(session_id)
    """)
    conn.execute(f"""
        GRANT USAGE ON SCHEMA public TO "{client_id}"
    """)
    conn.execute(f"""
        GRANT SELECT, INSERT, UPDATE ON TABLE session_messages TO "{client_id}"
    """)
    conn.execute(f"""
        GRANT USAGE, SELECT ON SEQUENCE session_messages_id_seq TO "{client_id}"
    """)
    conn.commit()

logger.info(f"Lakebase grants applied for SPN {client_id}.")
