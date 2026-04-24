# Databricks notebook source
import asyncio
import json

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from semantic_curator.config import get_env, load_config
from semantic_curator.mcp import create_mcp_tools

# enable nested asyncio for Databricks notebooks
nest_asyncio.apply()

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------
# create vector search mcp url
host = w.config.host
vector_search_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"

logger.info("Vector Search MCP URL:")
logger.info(vector_search_mcp_url)

# COMMAND ----------
# Connect to vector search mcp
vs_mcp_client = DatabricksMCPClient(server_url=vector_search_mcp_url, workspace_client=w)

# List available tools
vs_tools = vs_mcp_client.list_tools()

logger.info(f"Vector Search MCP Tools ({len(vs_tools)}):")
logger.info("=" * 80)
for tool in vs_tools:
    logger.info(f"Tool Name: {tool.name}")
    logger.info(f"Description: {tool.description}")
    if tool.inputSchema:
        logger.info(f"Parameters: {list(tool.inputSchema.get('properties', {}).keys())}")

# COMMAND ----------
# Search for papers related to "machine learning"
tool_name = f"{cfg.catalog}__{cfg.schema}__semantic_scholar_index"

search_result = vs_mcp_client.call_tool(
    tool_name, {"query": "machine learning and neural networks"}
)

logger.info("Search Results:")
logger.info("=" * 80)
for content in search_result.content:
    logger.info(content.text)

# COMMAND ----------
# Check if genie space is configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    logger.info("Genie MCP URL:")
    logger.info(genie_mcp_url)

    # Connect to genie mcp
    genie_mcp_client = DatabricksMCPClient(server_url=genie_mcp_url, workspace_client=w)

    # List available tools
    genie_tools = genie_mcp_client.list_tools()

    logger.info(f"Genie MCP Tools ({len(genie_tools)}):")
    logger.info("=" * 80)
    for tool in genie_tools:
        logger.info(f"Tool Name: {tool.name}")
        logger.info(f"Description: {tool.description}")
else:
    logger.warning("Genie space not configured in project_config.yml")
    logger.info(
        "To use genie tools, add 'genie_space_id' to your "
        "project_config.yml with the ID of your genie space"
    )

# COMMAND ----------
# Define MCP server URLs
mcp_urls = [f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"]


# Add genie if configured
if hasattr(cfg, "genie_space_id") and cfg.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}")

logger.info(f"Loading tools from {len(mcp_urls)} MCP servers...")

# Create tools
mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} tools from MCP servers")
logger.info("Available Tools:")
for i, tool in enumerate(mcp_tools, 1):
    logger.info(f"{i}. {tool.name}")

# COMMAND ----------
# Create a tools dictionary for easy access
tools_dict = {tool.name: tool for tool in mcp_tools}

vector_search_tool_name = f"{cfg.catalog}__{cfg.schema}__semantic_scholar_index"

if vector_search_tool_name in tools_dict:
    search_tool = tools_dict[vector_search_tool_name]

    # Execute the tool, only takes the query parameter
    result = search_tool.exec_fn(query="Deep learning architectures")

    logger.info("Search Results:")
    logger.info(result)

# COMMAND ----------
# View tool specifications

if mcp_tools:
    logger.info("Tool Specifications for LLM:")
    logger.info("=" * 80)

    for tool in mcp_tools[:2]:
        logger.info(f"Tool: {tool.name}")
        logger.info(json.dumps(tool.spec, indent=2))


# COMMAND ----------
def test_mcp_connection(mcp_url: str) -> bool:
    """
    Test if MCP server is accessible
    Args:
        mcp_url: URL of the MCP server
    Returns:
        True if connection is successful
    """
    try:
        client = DatabricksMCPClient(server_url=mcp_url, workspace_client=w)
        tools = client.list_tools()
        logger.info("Connected to MCP server")
        logger.info(f" URL: {mcp_url}")
        logger.info(f"Tools Available: {len(tools)}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        return False


# Test Vector Search MCP
logger.info("Testing vector search MCP:")
test_mcp_connection(vector_search_mcp_url)


# COMMAND ----------
class SimpleAgent:
    """A simple agent that can call tools in a loop"""

    def __init__(self, llm_endpoint: str, system_prompt: str, tools: list):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {tool.name: tool for tool in tools}
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        """Get tool specifications for the LLM"""
        return [tool.spec for tool in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name"""
        if tool_name not in self._tools_dict:
            raise ValueError(f"Tool {tool_name} not found")
        return self._tools_dict[tool_name].exec_fn(**args)

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Chat with agent, allowing tool calls"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _iteration in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() if self._tools_dict else None,
            )

            assistant_message = response.choices[0].message

            if assistant_message.tool_calls:
                # Add assistant message with tool calls (exclude unsupported fields)
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in assistant_message.tool_calls
                        ],
                    }
                )

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    logger.info(f"Calling tool: {tool_name}({tool_args})")

                    try:
                        result = self.execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        }
                    )
            else:
                return assistant_message.content

        return "Max iterations reached."


# Create agent with mcp tools
agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=(
        "You are a helpful research assistant. Use the available tools to "
        "search for papers and answer questions."
    ),
    tools=mcp_tools,
)

logger.info("Agent created with the following tools:")
for tool in agent._tools_dict.values():
    logger.info(f"- {tool.name}")

# COMMAND ----------
logger.info("Testing agent with MCP tools:")
logger.info("=" * 80)

response = agent.chat("Find papers about transformer architectures")
logger.info(f"Agent response: {response}")
