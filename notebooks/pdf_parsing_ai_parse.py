# Databricks notebook source
from loguru import logger 
from databricks.connect import DatabricksSession  

from semantic_curator.config import get_env, load_config
from semantic_curator.data_processor import DataProcessor

spark = DatabricksSession.builder.getOrCreate()
logger.info("Using Databricks Connect SparkSession")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

processor = DataProcessor(spark= spark, config=cfg)
logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}, Volume: {cfg.volume}")
processor.process_and_save()