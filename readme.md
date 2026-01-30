# 🔬 ScientificRAG: An Agentic RAG Research Assistant

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Getting Started](#4-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Data Ingestion](#data-ingestion)
    - [Running the Server](#running-the-server)
    - [Running the Frontend](#running-the-frontend)
5. [Usage](#5-usage)
6. [Project Structure](#6-project-structure)
7. [Future Enhancements](#7-future-enhancements)
8. [Contributing](#8-contributing)
9. [License](#9-license)
10. [Contact](#10-contact)

## 1. Project Overview

ScientificRAG is an advanced AI-powered Research Assistant designed to help users quickly and accurately retrieve and synthesize information from scientific papers. Built using LangChain, LangGraph, and Milvus, this project leverages an agentic architecture to provide nuanced, context-aware responses to complex scientific queries. It integrates external web search for real-time information and an arXiv dataset for deep academic insights, all accessible via a real-time WebSocket interface.

The core idea is to go beyond simple keyword matching, using an intelligent agent to understand the user's intent, break down complex questions, route to the most appropriate tools (internal scientific database or external web search), and then aggregate the findings into a comprehensive and coherent answer.

## 2. Features

* **Agentic Orchestration (LangGraph):** Employs a sophisticated graph-based agent that dynamically routes queries through different tools and processes based on the question's nature.
* **Hybrid Retrieval (Milvus):** Utilizes Milvus as a vector database for efficient semantic and keyword search (BM25) over a large corpus of scientific papers (e.g., arXiv metadata).
* **Context-Aware Query Rewriting:** Rewrites ambiguous or follow-up questions using conversational history to generate clear, self-contained queries for optimal retrieval.
* **Multi-Source Information Synthesis:** Integrates both internal knowledge (arXiv dataset) and real-time web search (Tavily API) to provide comprehensive answers.
* **Safety Guardrails:** Includes a preliminary guardrail to filter out unsafe or off-topic queries, ensuring responsible AI usage.
* **Real-time Interaction (WebSockets):** Offers a live, streaming experience for updates on the agent's thought process and final answers, ideal for dynamic frontend integration.
* **Modular and Extensible:** Designed with a clear separation of concerns, making it easy to add new tools, update models, or integrate different data sources.

## 3. Architecture

The system is built around a LangGraph-powered agent that acts as an intelligent router and orchestrator.

**High-Level Flow:**
1.  **User Query:** A user sends a query via the WebSocket frontend.
2.  **Safety Guardrail:** The initial check filters out harmful or irrelevant queries.
3.  **Chat Summarization:** If part of an ongoing conversation, the chat history is summarized for context.
4.  **Query Analysis & Rewriting:** Complex or ambiguous queries are rewritten into clear, self-contained sub-queries.
5.  **Tool Selection (Router):** Based on the rewritten query, the agent intelligently decides the best tool:
    * **`DIRECT_ANSWER`:** For simple, conversational queries.
    * **`WEB_SEARCH`:** For current events, recent statistics, or general knowledge.
    * **`ARXIV_SEARCH`:** For scientific, academic, or research-oriented questions, leveraging the Milvus database.
6.  **Tool Execution:** The selected tool (Milvus Retrieval or Web Search) fetches relevant information.
7.  **Sub-Agent Processing:** For `ARXIV_SEARCH`, a sub-agent further processes retrieved documents to extract precise answers.
8.  **Response Aggregation:** Answers from various tools/sub-agents are aggregated into a single, comprehensive final response.
9.  **Real-time Streaming:** Updates and the final answer are streamed back to the user via WebSockets.

Here's a visual representation of the LangGraph architecture:

http://googleusercontent.com/image_generation_content/0



*(Note: The `graph.png` placeholder shows the visual structure of your LangGraph agent. Ensure this image is updated with a current graph visualization.)*

## 4. Getting Started

Follow these instructions to set up and run the ScientificRAG project.

### Prerequisites

* Python 3.9+
* Docker (for Milvus)
* Access to Google Gemini API Key
* Access to Tavily API Key
* Basic understanding of `asyncio` and `websockets`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ScientificRAG.git](https://github.com/your-username/ScientificRAG.git)
    cd ScientificRAG
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv scientific_rag
    # On Windows
    .\scientific_rag\Scripts\activate
    # On macOS/Linux
    source scientific_rag/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Make sure your `requirements.txt` file includes all necessary libraries like `langchain-milvus`, `langchain-google-genai`, `langraph`, `tavily-python`, `datasets`, `websockets`, `uvicorn`, `pandas`, `pydantic`, `sentence-transformers`, `pymilvus` etc.)

4.  **Set up Milvus (using Docker):**
    ```bash
    docker-compose -f docker/milvus-standalone-docker-compose.yaml up -d
    ```
    Verify Milvus is running:
    ```bash
    docker ps
    ```
    You should see `milvus-standalone` and `zookeeper` containers running.

### Configuration

1.  **Create a `.env` file:**
    Create a file named `.env` in the root directory of your project.

2.  **Populate `.env` with API keys and settings:**
    ```ini
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    
    # LangChain Tracing (Optional but Recommended)
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
    LANGCHAIN_API_KEY="YOUR_LANGCHAIN_SMITH_API_KEY"
    LANGCHAIN_PROJECT="ScientificRAG-Project" # Replace with your project name

    # Milvus Connection Details
    MILVUS_URI="YOUR_MILVUS_CLOUD_URI_IF_APPLICABLE" # e.g., "https://{cluster_id}.{region}.vectordb.zillizcloud.com:19530"
    MILVUS_API="YOUR_MILVUS_CLOUD_API_KEY" # Only if using Milvus Cloud
    MILVUS_COLLECTION_NAME="arxiv_papers"
    MILVUS_VECTOR_FIELD="vector" # Default name for vector field
    
    # Proxy settings (if you are behind a corporate proxy)
    HTTP_PROXY=""
    HTTPS_PROXY=""

    # Model Settings
    GENERATION_MODEL="gemini-2.5-flash" # Or "gemini-pro", "gemini-1.5-flash" etc.
    ```
    * **Milvus:** If running Milvus locally via Docker Compose, you might not need `MILVUS_URI` and `MILVUS_API` in the `.env` if your `MilvusDB` class defaults to `localhost` and no token. However, it's good practice to have them defined. Adjust `src/vector_database/milvus_