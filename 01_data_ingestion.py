# Databricks notebook source
# MAGIC %md
# MAGIC  PHASE 1: SCHEME INGESTION & DATA LAYER

# COMMAND ----------

# MAGIC %pip install datasets sentence-transformers faiss-cpu gradio
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U fsspec huggingface_hub datasets
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install python-dotenv kaggle
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from datasets import load_dataset
from pyspark.sql import functions as F
import os
from dotenv import load_dotenv

# COMMAND ----------

# Set Kaggle credentials temporarily in the environment
# os.environ['KAGGLE_USERNAME'] = dbutils.secrets.get(scope="hackathon", key="kaggle_user")
# os.environ['KAGGLE_KEY'] = dbutils.secrets.get(scope="hackathon", key="kaggle_user_key")


# Dynamically construct the path to the .env file in the current workspace directory
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
workspace_dir = "/Workspace" + "/".join(notebook_path.split("/")[:-1])
env_path = f"{workspace_dir}/.env"

# Load the variables
load_dotenv(env_path) # load_dotenv("/Workspace/Users/you/app/.env")

# Safety Check
if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    raise ValueError("Missing Kaggle credentials. Please check your .env file.")

print(f"Authenticated as: {os.getenv('KAGGLE_USERNAME')}")


# COMMAND ----------

# DBTITLE 1,Download Kaggle dataset to local filesystem
!mkdir -p /tmp/raw_data/
!kaggle datasets download -d jainamgada45/indian-government-schemes --unzip -p /tmp/raw_data/

# COMMAND ----------

# DBTITLE 1,Verify the download
# Verify the download using shell command
!ls -lh /tmp/raw_data/

# COMMAND ----------

# Let's verify the exact filename extracted by Kaggle
print("Files in /tmp/raw_data/:", os.listdir("/tmp/raw_data/"))

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG IF NOT EXISTS nyaya_hackathon;
# MAGIC CREATE SCHEMA IF NOT EXISTS nyaya_hackathon.schemes_app;
# MAGIC USE CATALOG nyaya_hackathon;
# MAGIC USE SCHEMA schemes_app;

# COMMAND ----------

# 1. Ingest gov_myscheme from Kaggle
print("Loading gov_myscheme dataset...")
local_csv_path = "/tmp/raw_data/updated_data.csv" 
pdf = pd.read_csv(local_csv_path)

cols_to_drop = [col for col in pdf.columns if 'unnamed' in str(col).lower()]
pdf = pdf.drop(columns=cols_to_drop)

# 3. Convert to Spark DataFrame
df_raw = spark.createDataFrame(pdf)

# 4. Standardize column names (removes spaces, makes lowercase)
df_clean = df_raw.toDF(*[c.replace(' ', '_').lower() for c in df_raw.columns])


# 5. Handle Nulls for Concatenation
# If we don't fill nulls, concat_ws will destroy the entire string if one column is missing
text_columns = [
    "scheme_name", "details", "benefits", 
    "eligibility", "schemecategory", "tags", "level"
]
df_filled = df_clean.fillna("", subset=text_columns)

# COMMAND ----------

df_clean.columns

# COMMAND ----------

df_clean.limit(10).display()

# COMMAND ----------

df_filled.limit(10).display()

# COMMAND ----------

# 6. Engineer the Vector Search Context
# We explicitly order this from high-level categorization down to specific details
df_silver = df_filled.withColumn(
    "search_context", 
    F.concat_ws(
        " | ", 
        F.col("scheme_name"),
        F.col("schemecategory"),
        F.col("level"),
        F.col("tags"),
        F.col("details"),
        F.col("eligibility")
    )
)

# 7. Write to Delta Table (Databricks Lakehouse architecture)
print("Writing structured data to Delta Lake...")
(
    df_silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true") # Forces schema update if you rerun
    .saveAsTable("nyaya_hackathon.schemes_app.silver_schemes")
)

print(" Success: Delta Table 'silver_schemes' is live and optimized for RAG.")

# COMMAND ----------

# Verify the final schema and data
display(spark.table("nyaya_hackathon.schemes_app.silver_schemes").select("scheme_name", "search_context").limit(3))