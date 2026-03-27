# Databricks notebook source

import json
from datetime import datetime

import requests
from loguru import logger

# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType

from semantic_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)

try:
    cfg = load_config("project_config.yml", env)
    CATALOG = cfg.catalog
    SCHEMA = cfg.db_schema

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    logger.info(f"Using schema: {CATALOG}.{SCHEMA}")
except FileNotFoundError:
    CATALOG = spark.sql("SELECT current_catalog()").first()[0]
    SCHEMA = spark.sql("SELECT current_schema()").first()[0]
    logger.warning(
        "project_config.yml not found; falling back to current catalog/schema: "
        f"{CATALOG}.{SCHEMA}"
    )

TABLE_NAME = "semantic_scholar_papers"

# COMMAND ----------

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
HEADERS = {"Accept": "application/json"}
FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "publicationDate",
    "authors",
    "fieldsOfStudy",
    "publicationTypes",
    "venue",
    "url",
    "openAccessPdf",
    "externalIds",
    "citationCount",
    "referenceCount",
]

DEFAULT_QUERY = (
    '(llm OR "large language model" OR rag OR "retrieval augmented generation")'
)
DEFAULT_YEAR_RANGE = "2023-"
DEFAULT_BATCH_SIZE = 100
MAX_PAGES = 5
MAX_RESULTS = 500

# COMMAND ----------


def _stringify_json(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def fetch_semantic_scholar_papers(
    query: str = DEFAULT_QUERY,
    year_range: str = DEFAULT_YEAR_RANGE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_pages: int = MAX_PAGES,
    max_results: int = MAX_RESULTS,
) -> list[dict[str, object]]:
    params = {
        "query": query,
        "fields": ",".join(FIELDS),
        "year": year_range,
    }
    if batch_size:
        params["limit"] = min(batch_size, 1000)

    ingestion_timestamp = datetime.now().isoformat()
    next_token = None
    records = []
    total_available = None

    for page in range(1, max_pages + 1):
        request_params = dict(params)
        if next_token:
            request_params["token"] = next_token

        response = requests.get(
            BASE_URL,
            params=request_params,
            headers=HEADERS,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        if total_available is None:
            total_available = payload.get("total")
            logger.info(f"Semantic Scholar estimated total matches: {total_available}")

        batch = payload.get("data", [])
        logger.info(f"Semantic Scholar page {page}: retrieved {len(batch)} papers")

        for paper in batch:
            authors = [
                author.get("name")
                for author in paper.get("authors", [])
                if author.get("name")
            ]
            publication_date = paper.get("publicationDate")
            processed = None
            if publication_date:
                try:
                    processed = int(
                        datetime.fromisoformat(publication_date).strftime("%Y%m%d%H%M")
                    )
                except ValueError:
                    processed = None

            records.append(
                {
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "year": paper.get("year"),
                    "publication_date": publication_date,
                    "authors": authors,
                    "fields_of_study": paper.get("fieldsOfStudy") or [],
                    "publication_types": paper.get("publicationTypes") or [],
                    "venue": paper.get("venue"),
                    "url": paper.get("url"),
                    "open_access_pdf": _stringify_json(paper.get("openAccessPdf")),
                    "external_ids": _stringify_json(paper.get("externalIds")),
                    "citation_count": paper.get("citationCount"),
                    "reference_count": paper.get("referenceCount"),
                    "ingestion_timestamp": ingestion_timestamp,
                    "processed": processed,
                    "volume_path": None,
                }
            )

            if len(records) >= max_results:
                logger.info(f"Reached max_results={max_results}; stopping pagination")
                return records[:max_results]

        next_token = payload.get("token")
        if not next_token:
            break

    return records


logger.info("Fetching Semantic Scholar papers...")
papers = fetch_semantic_scholar_papers()
logger.info(f"Fetched {len(papers)} Semantic Scholar papers")

if papers:
    sample = papers[0]
    logger.info(f"Sample title: {sample['title']}")
    logger.info(f"Sample authors: {sample['authors']}")
    logger.info(f"Sample paper ID: {sample['paper_id']}")
    logger.info(f"Sample URL: {sample['url']}")
else:
    logger.warning("No papers were returned from Semantic Scholar")

# COMMAND ----------

schema = StructType(
    [
        StructField("paper_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("abstract", StringType(), True),
        StructField("year", LongType(), True),
        StructField("publication_date", StringType(), True),
        StructField("authors", ArrayType(StringType()), True),
        StructField("fields_of_study", ArrayType(StringType()), True),
        StructField("publication_types", ArrayType(StringType()), True),
        StructField("venue", StringType(), True),
        StructField("url", StringType(), True),
        StructField("open_access_pdf", StringType(), True),
        StructField("external_ids", StringType(), True),
        StructField("citation_count", LongType(), True),
        StructField("reference_count", LongType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),
        StructField("volume_path", StringType(), True),
    ]
)

df = spark.createDataFrame(papers, schema=schema)
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

(
    df.write.format("delta")
    .mode("overwrite")
    .option("mergeSchema", "true")
    .saveAsTable(table_path)
)

logger.info(f"Created Delta table: {table_path}")
logger.info(f"Records: {df.count()}")

# COMMAND ----------

papers_df = spark.table(table_path)

logger.info(f"Table: {table_path}")
logger.info(f"Total papers: {papers_df.count()}")
logger.info("Schema:")
papers_df.printSchema()

logger.info("Sample records:")
papers_df.select("paper_id", "title", "year", "citation_count").show(5, truncate=60)

# COMMAND ----------

logger.info("Papers by year:")
papers_df.groupBy("year").count().orderBy("year", ascending=False).show()

logger.info("Most cited papers:")
papers_df.select("title", "citation_count", "paper_id").orderBy(
    "citation_count", ascending=False
).show(5, truncate=60)
