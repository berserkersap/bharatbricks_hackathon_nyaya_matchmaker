import gradio as gr
import pandas as pd
import faiss
import os
import tempfile
import shutil
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from databricks.sdk import WorkspaceClient
from huggingface_hub import hf_hub_download

# 1. Initialize Databricks SDK Client (Auto-authenticates securely in Apps)
w = WorkspaceClient()

# 2. Define Source Volume Paths
VOLUME_DIR = "/Volumes/nyaya_hackathon/schemes_app/app_storage"
FAISS_VOL = f"{VOLUME_DIR}/scheme_index.bin"
MAPPING_VOL = f"{VOLUME_DIR}/scheme_mapping.csv"
MODEL_VOL = f"{VOLUME_DIR}/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Helper function to safely stream large files from Volume to App's local disk
def download_from_volume(volume_path, file_name):
    local_path = os.path.join(tempfile.gettempdir(), file_name)
    if not os.path.exists(local_path):
        print(f"Downloading {file_name} from Unity Catalog...")
        
        # 1. Call the download method directly (no 'with' statement)
        resp = w.files.download(volume_path)
        
        # 2. Write the contents using a single context manager for the local file
        with open(local_path, "wb") as f:
            # Use shutil to stream chunks to disk to prevent RAM spikes
            shutil.copyfileobj(resp.contents, f)
            
        # 3. Cleanly close the SDK response stream
        resp.contents.close()
        print(f"Downloaded {file_name} to {local_path}")
        
    return local_path

# 3. Pull Assets to Local Container Storage
# print("Fetching backend assets...")
# local_faiss_path = download_from_volume(FAISS_VOL, "scheme_index.bin")
# local_mapping_path = download_from_volume(MAPPING_VOL, "scheme_mapping.pkl")
# local_model_path = download_from_volume(MODEL_VOL, "model.gguf")
print("Fetching custom backend assets...")
local_faiss_path = download_from_volume(FAISS_VOL, "scheme_index.bin")
local_mapping_path = download_from_volume(MAPPING_VOL, "scheme_mapping.csv")

# FAST PATH: Download LLM directly from Hugging Face CDN
print("Downloading LLM from Hugging Face...")
local_model_path = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
print("Downloaded LLM from Hugging Face.")

# 4. Load Models and Index into Memory
print("Initializing AI components...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(local_faiss_path)
df_map = pd.read_csv(local_mapping_path)

llm = Llama(
    model_path=local_model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# 5. Core Application Logic
def match_user_to_schemes(user_profile):
    # Embed the user's situation
    query_vec = embedder.encode([user_profile], convert_to_numpy=True)
    
    # Retrieve top 2 most semantically relevant schemes
    distances, indices = index.search(query_vec.astype('float32'), k=2)
    
    final_output = f"## Nyaya-Sahayak: Your Scheme Matches\n\n"
    
    for i in indices[0]:
        scheme = df_map.iloc[i]
        scheme_name = scheme['scheme_name']
        eligibility = scheme['eligibility']
        benefits = scheme['benefits']
        
        prompt = f"""<|im_start|>system
You are a strict governance eligibility officer. Read the scheme criteria and the user profile. State if the user is eligible, why, and what the benefits are. Be brief.<|im_end|>
<|im_start|>user
Scheme: {scheme_name}
Criteria: {eligibility}
Benefits: {benefits}

User Profile: {user_profile}

Am I eligible?<|im_end|>
<|im_start|>assistant
"""
        # Run inference
        response = llm(prompt, max_tokens=200, stop=["<|im_end|>"])
        llm_reasoning = response["choices"][0]["text"].strip()
        
        final_output += f"### {scheme_name}\n**Analysis:** {llm_reasoning}\n---\n"
        
    return final_output

# 6. Gradio UI Layout
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🏛️ Nyaya-Matchmaker: Government Scheme Matchmaker")
    gr.Markdown("Describe your life situation (age, gender, state, occupation, income) in natural language, and AI will map you to the exact welfare schemes you qualify for.")
    
    with gr.Row():
        input_box = gr.Textbox(lines=4, placeholder="e.g., I am a 45-year-old female farmer from Tamil Nadu earning 50,000 a year...", label="Your Profile")
    
    submit_btn = gr.Button("Find My Schemes", variant="primary")
    output_box = gr.Markdown(label="Eligibility Results")
    
    submit_btn.click(fn=match_user_to_schemes, inputs=input_box, outputs=output_box)

# 7. Launch
if __name__ == "__main__":
    interface.launch()