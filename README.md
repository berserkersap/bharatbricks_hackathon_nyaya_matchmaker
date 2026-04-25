Repo created for bharat bricks iitm hackathon
# Nyaya-Sahayak: Government Scheme Matchmaker 🏛️

**Nyaya-Sahayak is a CPU-optimized, Databricks-native RAG application that matches rural citizens to government welfare schemes using natural language. It translates unstructured life situations into verifiable eligibility verdicts by treating scheme criteria as legal statutes.**

---

## 📐 Architecture Diagram

The system follows a modular Lakehouse-to-App architecture, ensuring data persistence and secure serverless serving.



1. **Ingestion Layer:** PySpark cleans and schema-enforces Kaggle data into **Delta Lake**.
2. **Storage Layer:** **Unity Catalog Volumes** store the FAISS index and the quantized GGUF model binaries.
3. **Application Layer:** **Databricks Apps** (Serverless) hosts the Gradio UI and local inference engine.

---

## 🚀 How to Run (Exact Commands)

To ensure the judges can reproduce the result, follow these steps in order.

### 1. Environment Configuration
* **Project Files:** Clone this repository into your Databricks Workspace (Repos).
* **Secrets:** Create a `.env` file in the project root with your Kaggle credentials:
  ```text
  KAGGLE_USERNAME=your_username
  KAGGLE_KEY=your_api_key
  ```
### 2. Infrastructure Setup (SQL)
* Run the following commands in a Databricks SQL Warehouse or Notebook to initialize your persistent storage:
```code
CREATE CATALOG IF NOT EXISTS nyaya_hackathon;
CREATE SCHEMA IF NOT EXISTS nyaya_hackathon.schemes_app;
CREATE VOLUME IF NOT EXISTS nyaya_hackathon.schemes_app.app_storage;
```

### 3. Pipeline Execution

Open the workspace and run the three notebooks in this specific order:

* 01_data_ingestion.ipynb: Handles the .env loading, Kaggle API download, and Delta Table creation.

* 02_build_vector_index.ipynb: Generates all-MiniLM-L6-v2 embeddings and saves the scheme_index.bin and scheme_mapping.csv to the Volume.

* 03_stage_model.ipynb: Downloads the Qwen2.5-1.5B-Instruct-GGUF model and copies it to the persistent Volume.

### 4. Deploying the Databricks App
1.  Navigate to the **Apps** section in the Databricks sidebar.
2.  Click **Create App** and select the folder containing `app.py` and `requirements.txt`.
3.  **Resource Configuration:** * Under the **Resources** tab, grant the app access to the `nyaya_hackathon` catalog.
    * Ensure the App has access to the **Unity Catalog Volume** where your models and index are stored.
4.  **Compute Settings:** Set the memory to **at least 12GB (Large)** to accommodate the LLM and Embedding models in-memory.
5.  **Environment Variables:** If you have any specific secrets or configuration keys, add them here.
6.  **Deploy:** Click **Deploy**. The platform will containerize your code and provide a secure URL.

---

## 🎥 Demo Steps
1.  Open the **Deployed Prototype Link** provided in the submission.
2.  **Sample Prompt:** Copy and paste the following into the text box:
    > *"I am a 45-year-old female farmer from Tamil Nadu earning 50,000 a year. My husband passed away recently and I need financial help."*
3.  **Action:** Click **"Find My Schemes"**.
4.  **Observation:** The system will perform a semantic search to retrieve relevant schemes (e.g., Widow Pensions or Agricultural Relief) and use the local LLM to provide a logic-based eligibility analysis.

---

## 🛠️ Tech Stack & Models
* **Databricks Technologies:** Unity Catalog (Volumes & Tables), Delta Lake, Databricks Apps, Databricks SDK, PySpark.
* **Open-Source Models:**
    * **LLM:** `Qwen2.5-1.5B-Instruct-GGUF` (4-bit quantization for CPU-only reasoning).
    * **Embeddings:** `all-MiniLM-L6-v2` (384-dimensional dense vectors).
    * **Vector Search:** `FAISS` (IndexFlatL2).

---

## 📝 Project Write-up
Built on the Databricks Lakehouse, this AI matchmaker translates unstructured citizen profiles into targeted welfare scheme eligibility. It utilizes Spark for ingestion into Delta Lake, distributed CPU-vectorization via FAISS, and a quantized local Qwen model for in-memory, hallucination-free reasoning. Designed strictly for low-compute environments without dedicated GPU endpoints, it proves that governance accessibility can be achieved efficiently using standard serverless CPU hardware.

---

## 🔗 Deployed Prototype
Link [https://nyaya-matchmaker-7474650366114452.aws.databricksapps.com/]
