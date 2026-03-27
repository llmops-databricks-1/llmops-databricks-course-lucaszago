"""
arXiv API
   ↓ (download_and_store_papers)
PDFs in Volume + arxiv_papers table
   ↓ (parse_pdfs_with_ai)
ai_parsed_docs_table (JSON)
   ↓ (process_chunks)
arxiv_chunks_table (clean text + metadata)
   ↓ (VectorSearchManager - separate class) (2.4 notebook)
Vector Search Index (embeddings)
"""

import json
import os
import re
import time
from datetime import datetime

import requests
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import (
    col,
    concat_ws,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from semantic_curator.config import ProjectConfig


class DataProcessor:
    """
    DataProcessor handles the complete workflow of:
    - Downloading papers from Semantic Scholar
    - Storing paper metadata
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.volume = config.volume

        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())
        self.pdf_dir = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}/{self.end}"
        os.makedirs(self.pdf_dir, exist_ok=True)
        self.papers_table = f"{self.catalog}.{self.schema}.semantic_scholar_papers"
        self.parsed_table = f"{self.catalog}.{self.schema}.ai_parsed_docs_table"

    def _get_range_start(self) -> str:
        """
        Get start time range for semantic scholar paper search.
        If semantic_scholar_papers table exists, uses max(processed) as start.
        Otherwise, uses 3 days ago as start.

        Returns:
            start string in "YYYYMMDDHHMM" format
        """

        if self.spark.catalog.tableExists(self.papers_table):
            result = self.spark.sql(f"""
                SELECT max(processed)
                FROM {self.papers_table}
            """).collect()
            start = str(result[0][0])
            logger.info(
                f"Found existing semantic_scholar_papers table. Starting from: {start}"
            )
        else:
            start = time.strftime("%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * 90))
            logger.info(
                f"No existing semantic_scholar_papers table. "
                f"Starting from 90 days ago: {start}"
            )
        return start

    # ------------------------------------------------------------------
    # Semantic Scholar API constants
    # ------------------------------------------------------------------
    _SS_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    _SS_HEADERS = {"Accept": "application/json"}
    _SS_FIELDS = [
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

    def _fetch_semantic_scholar_papers(
        self,
        query: str,
        pub_date_or_year: str,
        batch_size: int,
        max_results: int,
    ) -> list[dict]:
        """
        Paginate through the Semantic Scholar bulk-search endpoint.

        Args:
            query: Full-text search query string.
            pub_date_or_year: Date/year filter accepted by the API
                (e.g. "2024-01-15:" for "on or after that date").
            batch_size: Number of results per API page (max 1 000).
            max_results: Hard cap on total records returned.

        Returns:
            List of record dicts ready to be written to a Delta table.
        """
        params: dict = {
            "query": query,
            "fields": ",".join(self._SS_FIELDS),
            "publicationDateOrYear": pub_date_or_year,
            "limit": min(batch_size, 1000),
        }

        ingestion_timestamp = datetime.utcnow().isoformat()
        next_token: str | None = None
        records: list[dict] = []

        while True:
            request_params = dict(params)
            if next_token:
                request_params["token"] = next_token

            response = requests.get(
                self._SS_BASE_URL,
                params=request_params,
                headers=self._SS_HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()

            batch = payload.get("data", [])
            logger.info(
                f"Semantic Scholar: retrieved {len(batch)} papers "
                f"(total so far: {len(records) + len(batch)})"
            )

            for paper in batch:
                authors = [
                    a.get("name") for a in paper.get("authors", []) if a.get("name")
                ]
                pub_date = paper.get("publicationDate")
                processed: int | None = None
                if pub_date:
                    try:
                        processed = int(
                            datetime.fromisoformat(pub_date).strftime("%Y%m%d%H%M")
                        )
                    except ValueError:
                        processed = None

                records.append(
                    {
                        "paper_id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "year": paper.get("year"),
                        "publication_date": pub_date,
                        "authors": authors,
                        "fields_of_study": paper.get("fieldsOfStudy") or [],
                        "publication_types": paper.get("publicationTypes") or [],
                        "venue": paper.get("venue"),
                        "url": paper.get("url"),
                        "open_access_pdf": json.dumps(
                            paper.get("openAccessPdf"), ensure_ascii=False
                        )
                        if paper.get("openAccessPdf")
                        else None,
                        "external_ids": json.dumps(
                            paper.get("externalIds"), ensure_ascii=False
                        )
                        if paper.get("externalIds")
                        else None,
                        "citation_count": paper.get("citationCount"),
                        "reference_count": paper.get("referenceCount"),
                        "ingestion_timestamp": ingestion_timestamp,
                        "processed": processed,
                        "volume_path": None,
                    }
                )

                if len(records) >= max_results:
                    logger.info(f"Reached max_results={max_results}; stopping.")
                    return records[:max_results]

            next_token = payload.get("token")
            if not next_token:
                break

            # Respect Semantic Scholar rate limits (1 req/s without API key)
            time.sleep(1)

        return records

    def download_and_store_papers(
        self,
        query: str = (
            '(llm OR "large language model" OR rag OR "retrieval augmented generation")'
        ),
        max_results: int = 500,
        batch_size: int = 100,
    ) -> list[dict] | None:
        """
        Fetch papers from Semantic Scholar, download open-access PDFs,
        and store metadata in the semantic_scholar_papers Delta table.

        The method is idempotent: duplicate ``paper_id`` values are
        skipped via a MERGE statement.

        Args:
            query: Semantic Scholar full-text search query.
            max_results: Maximum number of papers to retrieve per run.
            batch_size: API page size (capped at 1 000 by the API).

        Returns:
            List of paper metadata dicts when at least one paper was
            downloaded, otherwise ``None``.
        """
        start = self._get_range_start()

        # YYYYMMDDHHMM → YYYY-MM-DD for the SS publicationDateOrYear filter
        pub_date_filter = f"{start[:4]}-{start[4:6]}-{start[6:8]}:"

        records = self._fetch_semantic_scholar_papers(
            query=query,
            pub_date_or_year=pub_date_filter,
            batch_size=batch_size,
            max_results=max_results,
        )

        if not records:
            logger.info("No new papers found.")
            return None

        # Download open-access PDFs to the volume
        for record in records:
            paper_id = record["paper_id"]
            if not paper_id:
                continue

            pdf_url = None

            # Try openAccessPdf URL first
            open_access_pdf = record.get("open_access_pdf")
            if open_access_pdf:
                try:
                    pdf_info = json.loads(open_access_pdf)
                    pdf_url = pdf_info.get("url") if pdf_info else None
                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback: construct URL from arXiv ID
            if not pdf_url:
                external_ids = record.get("external_ids")
                if external_ids:
                    try:
                        ext = json.loads(external_ids)
                        arxiv_id = ext.get("ArXiv")
                        if arxiv_id:
                            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
                    except (json.JSONDecodeError, TypeError):
                        pass

            if pdf_url:
                try:
                    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)
                    pdf_path = f"{self.pdf_dir}/{safe_id}.pdf"
                    resp = requests.get(pdf_url, timeout=30)
                    resp.raise_for_status()
                    with open(pdf_path, "wb") as fh:
                        fh.write(resp.content)
                    record["volume_path"] = pdf_path
                    logger.info(f"Downloaded PDF for {paper_id}")
                except Exception:
                    logger.warning(f"Could not download PDF for {paper_id}.")

            # Avoid hammering external servers
            time.sleep(1)

        downloaded = sum(1 for r in records if r.get("volume_path"))
        logger.info(
            f"Fetched {len(records)} papers; {downloaded} PDFs saved to {self.pdf_dir}"
        )

        # ── Spark schema ──────────────────────────────────────────────
        schema = T.StructType(
            [
                T.StructField("paper_id", T.StringType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("abstract", T.StringType(), True),
                T.StructField("year", T.LongType(), True),
                T.StructField("publication_date", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField("fields_of_study", T.ArrayType(T.StringType()), True),
                T.StructField("publication_types", T.ArrayType(T.StringType()), True),
                T.StructField("venue", T.StringType(), True),
                T.StructField("url", T.StringType(), True),
                T.StructField("open_access_pdf", T.StringType(), True),
                T.StructField("external_ids", T.StringType(), True),
                T.StructField("citation_count", T.LongType(), True),
                T.StructField("reference_count", T.LongType(), True),
                T.StructField("ingestion_timestamp", T.StringType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        metadata_df = self.spark.createDataFrame(records, schema=schema)

        # Create the table on first run (mode="ignore" is a no-op if it
        # already exists)
        metadata_df.write.format("delta").mode("ignore").saveAsTable(self.papers_table)

        # MERGE to avoid duplicates based on paper_id
        metadata_df.createOrReplaceTempView("new_ss_papers")
        self.spark.sql(f"""
            MERGE INTO {self.papers_table} target
            USING new_ss_papers source
            ON target.paper_id = source.paper_id
            WHEN NOT MATCHED THEN INSERT (
                paper_id, title, abstract, year, publication_date,
                authors, fields_of_study, publication_types, venue,
                url, open_access_pdf, external_ids, citation_count,
                reference_count, ingestion_timestamp, processed,
                volume_path
            ) VALUES (
                source.paper_id, source.title, source.abstract,
                source.year, source.publication_date, source.authors,
                source.fields_of_study, source.publication_types,
                source.venue, source.url, source.open_access_pdf,
                source.external_ids, source.citation_count,
                source.reference_count, source.ingestion_timestamp,
                source.processed, source.volume_path
            )
        """)
        logger.info(f"Merged {len(records)} records into {self.papers_table}")
        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.
        Reads from the current run's PDF directory.
        Falls back to scanning all volume PDFs if the current directory is empty.
        """
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
        """)

        volume_root = f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

        # Try current run dir first; fall back to full volume scan
        pdf_source = self.pdf_dir
        try:
            count = self.spark.sql(f"""
                SELECT COUNT(*) FROM READ_FILES(
                    "{self.pdf_dir}",
                    format => 'binaryFile'
                )
            """).collect()[0][0]
            if count == 0:
                pdf_source = volume_root
        except Exception:
            pdf_source = volume_root

        # Only parse PDFs not already in the parsed table
        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                _metadata.file_path AS path,
                ai_parse_document(content) AS parsed_content,
                {self.end} AS processed
            FROM READ_FILES(
                "{pdf_source}",
                format => 'binaryFile',
                pathGlobFilter => '*.pdf',
                recursiveFileLookup => 'true'
            )
            WHERE _metadata.file_path NOT IN (
                SELECT path FROM {self.parsed_table}
            )
        """)

        logger.info(f"Parsed PDFs from {self.pdf_dir} and saved to {self.parsed_table}")

    @staticmethod
    def _extract_chunks(
        parsed_content_json: str,
    ) -> list[tuple[str, str]]:
        """
        Extract text chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing
                parsed document structure

        Returns:
            List of tuples (chunk_id, content)
        """
        parsed_dict = json.loads(parsed_content_json)
        chunks = []

        for element in parsed_dict.get("document", {}).get("elements", []):
            if element.get("type") == "text":
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                chunks.append((chunk_id, content))

        return chunks

    @staticmethod
    def _extract_paper_id(path: str) -> str:
        """
        Extract paper_id from file path.

        Args:
            path: File path (e.g., "/path/to/paper_id.pdf")

        Returns:
            Paper ID extracted from the path
        """
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text.

        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean text chunks.
        Reads from ai_parsed_docs table and saves to
        semantic_scholar_chunks_table.
        """
        logger.info(
            f"Processing parsed documents from "
            f"{self.parsed_table} for end date {self.end}"
        )

        df = self.spark.table(self.parsed_table).where(f"processed = {self.end}")

        # Define schema for the extracted chunks
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        extract_paper_id_udf = udf(self._extract_paper_id, StringType())
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        metadata_df = self.spark.table(self.papers_table).select(
            col("paper_id"),
            col("title"),
            col("abstract"),
            concat_ws(", ", col("authors")).alias("authors"),
            concat_ws(", ", col("fields_of_study")).alias("fields_of_study"),
            col("year"),
            col("citation_count"),
        )

        # Explode parsed chunks, clean text, and join metadata
        chunks_df = (
            df.withColumn("paper_id", extract_paper_id_udf(col("path")))
            .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("paper_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id"),
            )
            .join(metadata_df, "paper_id", "left")
        )

        # Write to chunks table
        chunks_table = f"{self.catalog}.{self.schema}.semantic_scholar_chunks_table"
        chunks_df.write.mode("append").saveAsTable(chunks_table)
        logger.info(f"Saved chunks to {chunks_table}")

        # Enable Change Data Feed for downstream Vector Search sync
        self.spark.sql(f"""
            ALTER TABLE {chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {chunks_table}")

    def process_and_save(self) -> None:
        """
        Complete workflow: download papers, parse PDFs,
        and process chunks.
        """
        # Step 1: Download papers and store metadata
        records = self.download_and_store_papers()

        if records is None:
            logger.info("No new papers to process. Exiting.")
            return

        downloaded = sum(1 for r in records if r.get("volume_path"))
        has_unprocessed = False
        if downloaded == 0:
            # Check if there are existing PDFs that haven't been parsed yet
            if self.spark.catalog.tableExists(self.papers_table):
                unparsed_count = self.spark.sql(f"""
                    SELECT COUNT(*) FROM {self.papers_table}
                    WHERE volume_path IS NOT NULL
                """).collect()[0][0]
                has_unprocessed = unparsed_count > 0

            if not has_unprocessed:
                logger.info("No PDFs downloaded — skipping parsing and chunking.")
                return
            logger.info(
                f"No new PDFs, but found {unparsed_count} existing PDFs to parse."
            )

        # Step 2: Parse PDFs with ai_parse_document
        self.parse_pdfs_with_ai()
        logger.info("Parsed documents.")

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
