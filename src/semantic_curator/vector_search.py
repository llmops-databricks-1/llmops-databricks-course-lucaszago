import time

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from semantic_curator.config import ProjectConfig


def _is_missing_resource_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    return any(
        marker in error_msg for marker in ["not found", "404", "resource_does_not_exist"]
    )


class VectorSearchManager:
    """Manages vector search endpoints and indexes for semantic scholar papers"""

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: ProjectConfig object
            endpoint_name: Name of the vector search endpoint (uses config if None)
            embedding_model: Name of the embedding model endpoint (uses config if None)
            usage_policy_id: ID of the usage policy for the endpoint (optional)
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.semantic_scholar_index"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_or_get_index(self, wait_for_ready: bool = True) -> dict:  # noqa: ANN401
        """Create or get vector search index and ensure it's ready.

        Args:
            wait_for_ready: Wait for index to reach READY state (default: True)

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()
        source_table = f"{self.catalog}.{self.schema}.semantic_scholar_chunks_table"

        # Try to get existing index
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")

            # Check index status and wait if needed
            if wait_for_ready:
                self._wait_for_index_ready(max_wait_seconds=300)
            return index
        except Exception as e:
            if not _is_missing_resource_error(e):
                logger.warning(f"Error getting index: {e}")

        logger.info(f"Index {self.index_name} not found, creating new index...")

        # Try to create the index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")

            # Wait for index to be ready
            if wait_for_ready:
                self._wait_for_index_ready(max_wait_seconds=600)

            # Trigger initial sync
            logger.info("Triggering initial index sync...")
            index.sync()
            logger.info("✓ Initial sync triggered")

            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                # Index exists but get_index failed earlier (transient) — retry
                logger.info(
                    f"Index already exists, retrying to retrieve it: {self.index_name}"
                )
                time.sleep(2)  # Wait a bit for consistency
                index = self.client.get_index(index_name=self.index_name)
                if wait_for_ready:
                    self._wait_for_index_ready(max_wait_seconds=300)
                return index
            else:
                logger.error(f"Failed to create index: {e}")
                raise

    def _wait_for_index_ready(self, max_wait_seconds: int = 600) -> None:
        """Wait for vector search index to reach READY state.

        Args:
            max_wait_seconds: Maximum time to wait for index to be ready

        Raises:
            TimeoutError: If index doesn't become ready within max_wait_seconds
        """
        start_time = time.time()
        poll_interval = 10  # seconds

        while time.time() - start_time < max_wait_seconds:
            try:
                index = self.client.get_index(index_name=self.index_name)
                status = (
                    index.describe().get("status", {}).get("ready", False)
                    if hasattr(index, "describe")
                    else None
                )

                # Try alternative status check
                if status is None:
                    try:
                        details = self.client._client.get(f"/indexes/{self.index_name}")
                        status = details.get("status", {}).get("ready", False)
                    except Exception:  # noqa: E722
                        pass

                logger.info(f"Index status: ready={status}")

                if status:
                    logger.info(f"✓ Index is ready: {self.index_name}")
                    return

            except Exception as e:
                logger.debug(f"Checking index status: {e}")

            elapsed = time.time() - start_time
            logger.info(
                f"Waiting for index to be ready... ({int(elapsed)}s/{max_wait_seconds}s)"
            )
            time.sleep(poll_interval)

        logger.warning(
            f"Index did not reach READY state within {max_wait_seconds} "
            "seconds, proceeding anyway"
        )

    def sync_index(self) -> None:
        """Sync the vector search index with the source table."""
        logger.info(f"Syncing vector search index: {self.index_name}")
        index = self.create_or_get_index(wait_for_ready=True)
        index.sync()
        logger.info("✓ Index sync triggered")
        # Wait for sync to complete
        time.sleep(5)

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict | None = None,
        max_retries: int = 3,
    ) -> dict:
        """Search the vector index with retry logic.

        Args:
            query: Search query text
            num_results: Number of results to return
            filters: Optional filters to apply
            max_retries: Maximum number of retries on failure

        Returns:
            Search results dictionary
        """
        for attempt in range(max_retries):
            try:
                # Ensure index exists and is ready
                index = self.create_or_get_index(wait_for_ready=True)

                results = index.similarity_search(
                    query_text=query,
                    columns=["id", "text", "title", "paper_id"],
                    num_results=num_results,
                    filters=filters,
                )
                return results

            except Exception as e:
                error_msg = str(e)

                # Check if it's a "not found" error
                if _is_missing_resource_error(e):
                    logger.warning(
                        f"Index not found (attempt {attempt + 1}/{max_retries}): "
                        f"{error_msg}"
                    )

                    if attempt < max_retries - 1:
                        logger.info("Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
                    else:
                        logger.error(
                            f"Index still not found after {max_retries} attempts"
                        )
                        raise
                else:
                    # Other errors should not be retried
                    logger.error(f"Search failed: {error_msg}")
                    raise

        raise RuntimeError(f"Search failed after {max_retries} retries")
