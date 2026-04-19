# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Understanding embeddings
# MAGIC - Different embedding models
# MAGIC - Creating vector search endpoints
# MAGIC - Creating and syncing vector search indexes
# MAGIC - Querying with similarity search
# MAGIC - Advanced options: filters, hybrid search, reranking

# COMMAND ----------

from databricks.vector_search.reranker import DatabricksReranker
from databricks.sdk.runtime import dbutils
from loguru import logger
from pyspark.sql import SparkSession

from semantic_curator.config import get_env, load_config
from semantic_curator.utils.common import get_widget
from semantic_curator.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
run_id = get_widget("run_id", None)
is_job_run = bool(run_id and run_id != "local")

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")

# COMMAND ----------

# Create endpoint and index
logger.info("Setting up vector search index...")
index = vs_manager.create_or_get_index()

logger.info("\n✓ Vector search setup complete!")
logger.info(f"  Index: {vs_manager.index_name}")
logger.info(
    f"  Source: {vs_manager.catalog}.{vs_manager.schema}.semantic_scholar_chunks_table"
)
logger.info(f"  Embedding Model: {vs_manager.embedding_model}")

# Ensure index is synced with latest data
logger.info("\nEnsuring index is synced with latest data...")
try:
    vs_manager.sync_index()
    logger.info("✓ Index sync complete")
except Exception as e:
    error_msg = str(e).lower()
    if "not ready" in error_msg or "initializing" in error_msg:
        logger.warning(
            "Vector search index is still initializing. "
            "Provisioning completed; rerun this step later to force sync."
        )
    else:
        raise

if is_job_run:
    logger.info("Job context detected. Skipping demo searches after setup.")
    dbutils.notebook.exit("Vector search setup completed")

# COMMAND ----------


def parse_vector_search_results(results: dict) -> list[dict]:
    """Parse vector search results from array format to dict format."""
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])
    return [dict(zip(columns, row_data, strict=False)) for row_data in data_array]


# COMMAND ----------

# Simple similarity search
query = "What are the latest techniques in machine learning?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "paper_id"],
    num_results=5,
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. Paper: {row.get('title', 'N/A')}")
    logger.info(f"   Paper ID: {row.get('paper_id', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
    logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# Search with metadata filters
query = "neural networks and deep learning"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "year", "authors"],
    filters={"year": "2026"},
    num_results=3,
)

logger.info(f"Query: {query}")
logger.info("Filter: year = 2026\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Year: {row.get('year', 'N/A')}")
    authors = row.get("authors", "N/A")
    logger.info(f"   Authors: {str(authors)[:100]}...")
    logger.info(f"   Text: {row.get('text', '')[:150]}...")

# COMMAND ----------

# Hybrid search example
query = "transformer architecture attention mechanism"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title"],
    num_results=5,
    query_type="hybrid",
)

logger.info(f"Query: {query}")
logger.info("Search Type: Hybrid (Semantic + Keyword)\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}...")

# COMMAND ----------

# Hybrid search with reranking
query = "large language models for code generation"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "abstract"],
    num_results=5,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title", "abstract"]),
)

logger.info(f"Query: {query}")
logger.info("With reranking on: text, title, abstract\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(parse_vector_search_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Abstract: {row.get('abstract', '')[:150]}...")
    logger.info(f"   Text: {row.get('text', '')[:150]}...")

# COMMAND ----------

# Compare search strategies
query = "attention mechanisms in transformers"

logger.info(f"Query: {query}\n")

# Strategy 1: Basic semantic search
results_basic = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
)

logger.info("Strategy 1: Basic Semantic Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_basic), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# Strategy 2: Hybrid search
results_hybrid = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid",
)

logger.info("\nStrategy 2: Hybrid Search")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_hybrid), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")

# Strategy 3: Hybrid + Reranking
results_reranked = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title"]),
)

logger.info("\nStrategy 3: Hybrid + Reranking")
logger.info("-" * 80)
for i, row in enumerate(parse_vector_search_results(results_reranked), 1):
    logger.info(f"{i}. {row.get('title', 'N/A')[:60]}...")


# Check index status
index_info = vs_manager.client.get_index(
    endpoint_name=vs_manager.endpoint_name, index_name=vs_manager.index_name
)

logger.info("Index Information:")
logger.info(f"  Name: {index_info.name}")
logger.info(f"  Endpoint: {index_info.endpoint_name}")
