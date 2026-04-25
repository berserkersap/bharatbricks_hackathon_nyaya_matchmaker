# Databricks notebook source
# MAGIC %pip install sentence-transformers faiss-cpu
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer

# COMMAND ----------

# 1. Pull the Delta Table into driver memory as the data is very small
print("Loading 'silver_schemes' from Unity Catalog...")
df_schemes = spark.table("nyaya_hackathon.schemes_app.silver_schemes").toPandas()

# COMMAND ----------

# 2. Load the CPU-optimized embedding model
# all-MiniLM-L6-v2 is incredibly fast, takes ~90MB of RAM, and outputs 384D vectors
print("Loading MiniLM embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# COMMAND ----------

# 3. Generate Embeddings
print(f"Vectorizing {len(df_schemes)} scheme contexts...")
# Convert to numpy array as required by FAISS
contexts = df_schemes['search_context'].tolist()
embeddings = embedder.encode(contexts, convert_to_numpy=True, show_progress_bar=True)

# COMMAND ----------

# 4. Build the FAISS Index
# We use IndexFlatL2 for exact L2 distance search (perfect for small datasets)
dimension = embeddings.shape[1]  # Should be 384
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# COMMAND ----------

# 5. Create storage directory in Volumes (serverless-compatible)
volume_path = "/Volumes/nyaya_hackathon/schemes_app/app_storage"
os.makedirs(volume_path, exist_ok=True)

# 6. Save the Index and Mapping to Volumes
# The App (Phase 3) will load these two files into RAM when a user connects
faiss_path = os.path.join(volume_path, "scheme_index.bin")
mapping_path = os.path.join(volume_path, "scheme_mapping.pkl")

faiss.write_index(index, faiss_path)

# Save the DataFrame to easily retrieve the exact scheme details (benefits, application) later
df_schemes.to_pickle(mapping_path)

print(f"Success: Assets saved to Volumes.")
print(f"Index: {faiss_path}")
print(f"Mapping: {mapping_path}")