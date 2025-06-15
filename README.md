
# 🎸 RAG-n-Rock: RAG-Powered Chat Application

  

**RAG-n-Rock** is a lightweight, production-ready **Retrieval-Augmented Generation (RAG)** chat application built using FastAPI, LangChain, ChromaDB, and Ollama. It enables users to upload files, ask questions, and get context-aware answers powered by local LLMs.

  

---

  

## 🔍 Overview

  

This project provides:

  

- ✅ **File Upload & Ingestion**: Supports `.pdf`, `.docx`, `.txt`, `.csv`, `.xlsx`. Files are parsed, chunked, and stored in a vector database.

- ✅ **Chat Interface**: Query documents using hybrid retrieval — combining keyword matching, metadata filtering, and semantic search.

- ✅ **User Authentication**: Secure login and session management.

- ✅ **Admin Features**: Clear all data with admin token protection.

- ✅ **Health Monitoring**: Health check endpoint for status and service readiness.

- ✅ **Structured Logging**: Centralized logging across all modules for debugging and auditing.

  

---

  

## 🧩 Technologies Used

| Component    | Technology                                                                 |
|--------------|---------------------------------------------------------------------------|
| Framework    | [FastAPI](https://fastapi.tiangolo.com/)                                  |
| ORM          | [SQLAlchemy](https://www.sqlalchemy.org/)                                 |
| Vectorstore  | [ChromaDB](https://docs.trychroma.com/)                                   |
| LLM          | [Ollama](https://ollama.ai/) via [LangChain](https://python.langchain.com/) |
| Embeddings   | `nomic-embed-text`                                                        |
| Auth         | JWT-based OAuth2                                                          |
| Logging      | Structured, centralized logging                                           |

  

## 🛠️ Setup & Installation

  

### Prerequisites

  

- Python 3.9+

- Ollama

- SQLite or PostgreSQL (SQLite used by default)

  

### Steps

  

1.  **Start Ollama**

2.  **Install Dependencies**
```bash
pip install -r requirements.txt
```
  3.  **Run the App**
```bash
uvicorn main:app --reload
```
4.  **Access API Docs**
```bash
http://localhost:8000/docs
```
  

## 🚀 Features

- 📁 Upload Files : Upload supported documents and store them securely in the vectorstore.

- 💬 Chat with Files : Ask natural language questions and get answers derived from your uploaded files.

- 🔍 Hybrid Retrieval : Combines semantic search, keyword matching, and metadata filtering.

- 👥 Multi-user Support : Each user only sees their own files and chat history.

- 🛡️ Admin Endpoint : Delete all files and chat history with an admin token.

- 📊 Health Check : Monitor the health of the database, vectorstore, and LLM

  

## 🔐 Security

- JWT-based authentication

- Password hashing using bcrypt

- Admin-token protected destructive actions

- Input validation and sanitization.