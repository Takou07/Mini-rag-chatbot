# Mini-rag-chatbot
Chatbot RAG minimal basé sur LangChain, FAISS et Gradio.

### 🧠 **Project Description — Mini RAG Chatbot**

> A lightweight chatbot powered by **Retrieval-Augmented Generation (RAG)**, capable of answering questions based on the content of a **single document (PDF, article, or wiki page)**.
> The project demonstrates how to build a minimal RAG pipeline using **LangChain**, **FAISS**, **HuggingFace embeddings**, and a **LLM (GPT-4o-mini)** — perfect for understanding how modern AI assistants combine search and reasoning.

---

### ⚙️ **How It Works**

1. **Document Ingestion** – Loads and preprocesses a document (e.g. PDF).
2. **Text Splitting** – Breaks it into small text chunks for semantic indexing.
3. **Embedding Generation** – Transforms each chunk into vector representations using `sentence-transformers`.
4. **Vector Search (FAISS)** – Retrieves the most relevant chunks based on the user’s query.
5. **LLM Response** – Combines retrieved context with the question to generate a natural language answer.

---

### 💡 **Why This Project**

This project was built as part of the **“Pédiluve” maker challenge** — a mini hackathon designed to test motivation, autonomy, and clarity.
It shows how to:

* Build an end-to-end AI prototype quickly
* Use **retrieval + generation** to ground answers in real data
* Create a simple, functional **Gradio interface** to interact with the model

---

### 🧰 **Tech Stack**

* 🧱 **LangChain** – Orchestration and chaining logic
* 🔍 **FAISS** – Local vector database for similarity search
* 🧩 **HuggingFace Embeddings** – Semantic encoding of text
* 🤖 **OpenAI GPT-4o-mini** – Language model for response generation
* 🖥️ **Gradio** – Minimal UI for chat interaction

---

### 🚀 **Main Features**

* Query any PDF or local document interactively
* Local vector search (no external DB needed)
* Fast, light, and privacy-friendly
* Clean architecture — easy to extend for multi-source or API-based retrieval

---

### 🎯 **Learning Outcomes**

By completing this project, I will learn to:

* Implement a **RAG pipeline from scratch**
* Understand **how retrieval improves factual accuracy**
* Integrate **LLMs with external knowledge sources**
* Build and deploy a simple **AI chatbot interface**



