# 🎥 YouTube Video Q&A Assistant & Dual-Brain Psychology Chatbot

A multi-purpose advanced AI portfolio repository showcasing state-of-the-art context-aware RAG systems, Large Language Model orchestration, and framework integrations across two main intelligent applications.

---

## 📺 Application 1: YouTube Video Q&A Assistant (RAG Chatbot)
A production-ready, context-aware RAG (Retrieval-Augmented Generation) application built with Python. This application allows users to input any YouTube video URL, automatically extracts its transcript, chunks and embeds the data locally using HuggingFace models, and indexes it into a high-performance FAISS vector store. Users can then dynamically chat with the video's content using Llama-3 with full session state conversational memory.

### 🎥 Live Demo (YouTube Q&A)
<video src="demo.mov" width="100%" controls></video>

### 🚀 Key Features
- **Instant Transcript Extraction:** Leverages `youtube-transcript-api` to pull raw video data dynamically.
- **Local RAG Pipeline:** Implements optimized text splitting, chunking, and text embeddings via HuggingFace's Sentence-Transformers.
- **In-Memory Vector Indexing:** Utilizes a fast local FAISS database for lightning-quick query retrieval.
- **Advanced LLM Orchestration:** Built using LangChain frameworks and LCEL (LangChain Expression Language) workflows.
- **Llama-3 Integration:** Connects to the state-of-the-art Meta Llama-3-8B-Instruct model.
- **Intuitive UI & Memory:** Built with Streamlit, featuring sleek interactive chat speech bubbles and persistent `st.session_state` conversational memory management.

---

## 🧠 Application 2: Dual-Brain Psychology Chatbot
A Python-based chatbot that helps users explore their emotional patterns using the Dual Brain Psychology (DBP) framework. This project supports both command-line (CLI) and API (FastAPI) usage, integrates LLMs (OpenAI/Google), and provides entity extraction and brain profile analysis.

### 🚀 Key Features
- **Empathetic DBP Dialogue:** All bot responses are grounded in Fred Schiffer’s Dual Brain Psychology, avoiding clinical or diagnostic language.
- **Named Entity Recognition:** Uses spaCy to extract entities from user messages.
- **LLM Integration:** Supports OpenAI (GPT-4o, GPT-3.5-turbo) and Google Generative AI via LangChain.
- **Knowledge Base Retrieval:** Uses FAISS vector store for context-aware responses.
- **Session Management:** Maintains per-user chat history and brain profile analysis (CLI and API).
- **Data Logging:** Saves chat sessions and brain profile summaries to JSON files in `data/`.
- **API & CLI:** Interact via REST API or command-line.

---

## 🛠️ Project Structure

langchainmodels/
│
├── youtube_video_q&a_streamlit_app.py   # Core Streamlit Frontend & RAG Controller
├── rag_system.py                       # LangChain Retrieval & Prompt Pipeline Logic
├── demo.mov                            # 1-Minute Live Demo Execution Video
│
├── Chat_Bot/                           # Dual-Brain Psychology Chatbot Subsystem
│   ├── app.py                          # FastAPI server (REST API)
│   ├── main.py                         # CLI entry point
│   ├── conversation.py                 # Chat session logic, history, greeting/goodbye
│   ├── entity_extractor.py             # spaCy NER and brain profile analysis
│   ├── brain_assessor.py               # Brain dominance/emotion analysis helpers
│   ├── llm_handler.py                  # LLM (OpenAI/Google) interface
│   ├── prompts.py                      # System, greeting, and goodbye prompts
│   ├── config.py                       # Environment/config management
│   ├── loader.py                       # Knowledge base loading utilities
│   ├── vector_store.py                 # FAISS vector store setup and retrieval
│   ├── render.yaml                     # Render.com deployment config
│   └── data/                           # Knowledge files and session logs
│
├── requirements.txt                    # Combined Managed Python Dependencies
├── .env                                # Secure Environment API Token Store
└── doc/                                # Visualizing Assets & Infographics
└── RAG_Architecture_Slides.pdf    # Multi-Asset Technical Slide Deck

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/Mrup1/langchain-portfolio.git](https://github.com/Mrup1/langchain-portfolio.git)
cd langchainmodels


2. Configure Your Environment
It is highly recommended to use a Python virtual environment:



python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


3. Add Credentials
Create a .env file in the root directory and securely add your tokens and API keys:

HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-genai-key


4. Prepare Knowledge Base (For DBP Chatbot)
Place your knowledge/text files in the Chat_Bot/data/ directory. Update paths in config.py if needed.

🎯 Usage
Running Application 1: YouTube Video Q&A (Streamlit)

streamlit run youtube_video_q&a_streamlit_app.py

Open your local browser to the address provided in the terminal (typically http://localhost:8501), drop in a video URL, let it process, and begin asking questions!


Running Application 2: Dual-Brain Psychology Chatbot
CLI Mode


python Chat_Bot/main.py


The bot will greet you and you can start chatting.
Type bye to end the session. Session summaries are saved in Chat_Bot/data/.


uvicorn Chat_Bot.app:app --reload


POST /chat — Start or continue a chat session. JSON body:
{ "user_id": "string", "message": "string" }
    ```
- **Session End:** Send message `"bye"` to end and save the session in the data folder.

---

## 🚀 Deployment (DBP Chatbot Subsystem)
- **Render.com:** Use `render.yaml` for deployment. The app will start on port 10000.
- **Custom:** Use `uvicorn Chat_Bot.app:app --host 0.0.0.0 --port <PORT>` for self-hosting.

---

## 🔧 Troubleshooting
- For dependency issues, recreate your virtual environment and reinstall requirements.
- Ensure all API keys are correctly set in the `.env` file.
- If spaCy model errors occur for the DBP Chatbot, run:
```bash
  python -m spacy download en_core_web_sm


For vector store errors, ensure knowledge files exist in their respective directories.


🛡️ License


Distributed under the MIT License. See LICENSE for more information.
