# LLM Financial Analyst

## Description

This project is an AI-powered application designed to analyze financial reports (10-K, 10-Q, earnings calls, news) using **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** techniques.  
The goal is to generate actionable insights and summaries for risk analysis and decision-making.

The project combines:
- Extraction and preprocessing of financial data (PDFs, Yahoo Finance API, SEC Edgar)  
- Text vectorization with **embeddings**  
- Contextual search using **FAISS**  
- Response generation using a local LLM (**Mistral 7B Instruct via Ollama**)  
- A simple web interface with **Streamlit**  

---

## Tech Stack

- **Language & Environment**: Python 3.11+, virtual environment (venv)  
- **Data Extraction**: `pdfplumber`, `yfinance`, SEC Edgar  
- **LLM & NLP**: `transformers`, `sentence-transformers`, `huggingface_hub`, `FAISS`  
- **Interface**: `Streamlit`  
- **Local LLM**: [Ollama](https://ollama.ai/) + Mistral 7B Instruct  

---

## Project Structure

```text
FinanceLLM/
├── envs/finance-venv/       # Python virtual environment
├── data/                    # Storage for PDF reports and financial data
├── src/                     # Python scripts
│   ├── extract.py           # PDF extraction and preprocessing
│   ├── embeddings.py        # Embeddings generation and storage
│   ├── rag.py               # Contextual search and RAG pipeline
│   └── llm_query.py         # Interface with LLM (Ollama)
├── app.py                   # Streamlit UI
├── requirements.txt         # Python dependencies
└── README.md
