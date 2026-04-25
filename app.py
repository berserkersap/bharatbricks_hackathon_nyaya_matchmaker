import gradio as gr
import pandas as pd
import faiss
import os
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# 1. Define Volume Paths
VOLUME_PATH = "/Volumes/nyaya_hackathon/schemes_app/app_storage"
FAISS_PATH = os.path.join(VOLUME_PATH, "scheme_index.bin")
MAPPING_PATH = os.path.join(VOLUME_PATH, "scheme_mapping.pkl")
MODEL_PATH = os.path.join(VOLUME_PATH, "qwen2.5-1.5b-instruct-q4_k_m.gguf")

# 2. Load the Embedding Model & Vector Search (Happens once on startup)
print("Loading embeddings and index...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(FAISS_PATH)
df_map = pd.read_pickle(MAPPING_PATH)

# 3. Load the Quantized LLM (CPU Optimized)
print("Loading LLM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,       # Context window
    n_threads=4,      # Restrict threads to prevent CPU thrashing
    verbose=False
)

def match_user_to_schemes(user_profile):
    """Embeds the user profile, finds schemes, and verifies eligibility."""
    
    # Generate vector for the user's situation
    query_vec = embedder.encode([user_profile], convert_to_numpy=True)
    
    # Retrieve top 2 most semantically relevant schemes
    distances, indices = index.search(query_vec.astype('float32'), k=2)
    
    final_output = f"## Nyaya-Sahayak: Your Scheme Matches\n\n"
    
    for i in indices[0]:
        scheme = df_map.iloc[i]
        scheme_name = scheme['scheme_name']
        eligibility = scheme['eligibility']
        benefits = scheme['benefits']
        
        # Strict prompt engineering to prevent hallucination
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

# 4. Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🏛️ Nyaya-Sahayak: Government Scheme Matchmaker")
    gr.Markdown("Describe your life situation (age, gender, state, occupation, income) in natural language, and AI will map you to the exact welfare schemes you qualify for.")
    
    with gr.Row():
        input_box = gr.Textbox(lines=4, placeholder="e.g., I am a 45-year-old female farmer from Tamil Nadu earning 50,000 a year...", label="Your Profile")
    
    submit_btn = gr.Button("Find My Schemes", variant="primary")
    output_box = gr.Markdown(label="Eligibility Results")
    
    submit_btn.click(fn=match_user_to_schemes, inputs=input_box, outputs=output_box)

# 5. Launch
if __name__ == "__main__":
    # Databricks Apps require binding to 0.0.0.0 and a specific port (usually 8080)
    interface.launch(server_name="0.0.0.0", server_port=8080)