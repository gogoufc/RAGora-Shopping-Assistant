# 🛒 Amazon C4 Agentic Shopping Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green)
![LLM](https://img.shields.io/badge/LLM-GLM--4.5--Air-purple)

> **Master's Thesis Project**: Business Analytics & Data Science, University of Macedonia.

An intelligent, agentic shopping assistant designed to navigate the massive **Amazon C4 Dataset** (1M+ products). Unlike simple chatbots, this system uses a **Supervisor-Worker architecture** via **LangGraph** to dynamically decide when to search for products and when to engage in casual conversation, maintaining memory throughout the session.

---

## ✨ Key Features

* **🧠 Agentic Orchestration:** Built on **LangGraph**, employing a **Supervisor Node** that intelligently routes queries between a `Search_Agent` and a `ChitChat_Agent`.
* **🔍 Semantic Product Discovery:** Implements **RAG (Retrieval-Augmented Generation)** using dense vector retrieval (FAISS) to find relevant products based on meaning, not just keywords.
* **💾 Multi-turn Memory:** Keeps context across the conversation (e.g., "Show me cheaper options") using `MemorySaver`.
* **🚀 High-Performance Indexing:** Utilizes `sentence-transformers/all-MiniLM-L6-v2` for generating lightweight yet effective embeddings (~384d).
* **💸 Cost-Effective Intelligence:** Optimized to run with **GLM-4.5-Air** (via OpenRouter), proving that high-quality agentic workflows don't require expensive proprietary models.

---

## 🏗️ System Architecture

The system moves beyond simple linear chains. It uses a stateful graph where a **Supervisor** acts as the brain:

1. **User Input** enters the graph.
2. **Supervisor Node** analyzes intent:
   * *Is this a product query?* → Route to **Search Tool**.
   * *Is this a greeting/follow-up?* → Route to **Chit-Chat**.
3. **Tools:**
   * `tools_c4.py`: Performs vector similarity search on the FAISS index.
4. **Response:** The LLM synthesizes the retrieved data into a helpful answer.

### 📂 Project Structure

```text
.
├── src/
│   ├── agent_graph.py    # 🧠 Core Logic: LangGraph Supervisor & Workflow
│   ├── chat_app.py       # 💬 Interface: CLI / Chat loop
│   ├── tools_c4.py       # 🛠️ Tools: Product search logic & Formatting
│   └── embeddings_c4.py  # ⚙️ ETL: Embedding generation & FAISS indexing
├── data/                 # 🗄️ Data Storage (gitignored usually)
│   ├── faiss_index_c4.bin
│   └── c4_metadata.pkl
├── .env                  # 🔑 API Keys configuration
└── requirements.txt      # 📦 Dependencies
```

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM** | GLM-4.5-Air | Accessed via OpenRouter API (Free tier friendly) |
| **Orchestration** | LangGraph | State management and cyclic graph workflows |
| **Embeddings** | all-MiniLM-L6-v2 | Sentence Transformers model for vectorization |
| **Vector Store** | FAISS | CPU-optimized similarity search for 1M+ vectors |
| **Dataset** | Amazon C4 | Large-scale e-commerce product dataset |

## 🚀 Quick Start

Follow these steps to set up the agent locally.

### 1. Clone the Repository
```bash
git clone https://github.com/gogoufc/RAGora-Shopping-Assistant.git
cd amazon-c4-agent
```

### 2. Set up Environment
Choose one of the following methods to set up your Python environment:

**Option A: Using venv (Standard)**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**Option B: Using Conda (Recommended)**

```bash
# Create a new environment with Python 3.10
conda create -n c4_agent python=3.10

# Activate the environment
conda activate c4_agent
```

### 3. Install Dependencies
Regardless of the environment you chose above, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the root directory. You can copy the example file, or create a new one:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LANGSMITH_API_KEY=your-key-here  # Optional: For tracing
```

### 5. Generate/Verify Embeddings
> **Note:** The FAISS index is required for the search tool to function.

Check if `data/faiss_index_c4.bin` exists. If not, run the generator (**Warning:** Intensive process):

```bash
cd src
python embeddings_c4.py
# ⏳ Expect ~8 hours on standard CPU for the full dataset
```

### 6. Run the Agent
Once the index is ready and your environment is active, start the chat interface:

```bash
python chat_app.py
```
## 💡 Usage Example

Once the app is running, you can interact naturally:

> **You:** "I need a gift for my brother, he is always losing his keys around the house."
>
> **Agent:** *(Routes to Search Tool...)*
> Found relevant items:
> 1. **Tile Mate Tracker**
>    *A great Bluetooth tracker to help him find his keys quickly.*
> 2. **Apple AirTag**
>    *Perfect if he uses the Apple ecosystem.*
>
> **You:** "Which one has better compatibility if he uses an Android phone?"
>
> **Agent:** *(Uses Memory & Chit-Chat...)*
> Since he uses an Android phone, the Tile Mate is the better option because AirTags are designed primarily for Apple devices...

## ⚠️ Limitations & Notes

* **Rate Limits:** The free tier of OpenRouter (GLM-4.5) may hit rate limits (Error 429) during intense testing.
* **Hardware:** Generating embeddings for 1M products requires approx. **2GB RAM** and meaningful CPU time.
* **Data Freshness:** The Amazon C4 dataset is static; prices and availability are historical.

## 📄 License

This project is part of an academic thesis.
