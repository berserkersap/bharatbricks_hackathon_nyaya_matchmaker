# Databricks notebook source
from huggingface_hub import hf_hub_download
import shutil
import os

# Download to local temp directory first (serverless-compatible)
local_model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    local_dir="/tmp/model_download"
)

# Copy to Volume for persistent storage
volume_path = "/Volumes/nyaya_hackathon/schemes_app/app_storage"
os.makedirs(volume_path, exist_ok=True)
final_model_path = os.path.join(volume_path, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
shutil.copy2(local_model_path, final_model_path)

print(f"Model saved to: {final_model_path}")