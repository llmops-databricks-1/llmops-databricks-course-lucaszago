import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from semantic_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

chunks_df = spark.table(f"{catalog}.{schema}.semantic_scholar_chunks_table")

logger.info(f"Total chunks available: {chunks_df.count()}")
chunks_df.show(5, truncate=50)


# Calculate chunk statistics
chunk_stats = chunks_df.select(
    F.avg(F.length(F.col("text"))).alias("avg_length"),
    F.min(F.length(F.col("text"))).alias("min_length"),
    F.max(F.length(F.col("text"))).alias("max_length"),
    F.count("*").alias("total_chunks"),
).collect()[0]

logger.info("Chunk Statistics:")
logger.info(f"  Total chunks: {chunk_stats['total_chunks']}")
logger.info(f"  Average length: {chunk_stats['avg_length']:.0f} characters")
logger.info(f"  Min length: {chunk_stats['min_length']} characters")
logger.info(f"  Max length: {chunk_stats['max_length']} characters")


def sentence_chunking(text: str, max_sentences: int = 5) -> list[str]:
    """Create chunks based on sentence boundaries.

    Args:
        text: Text to chunk
        max_sentences: Maximum sentences per chunk

    Returns:
        List of text chunks
    """
    # Simple sentence splitter (can be improved with spaCy/NLTK)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_sentences:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


sample_text = chunks_df.select("text").first()["text"]
sentence_chunks = sentence_chunking(sample_text, max_sentences=5)

logger.info(f"Number of sentence-based chunks: {len(sentence_chunks)}")
logger.info("\nFirst chunk preview:")
logger.info(sentence_chunks[0][:200] + "...")
