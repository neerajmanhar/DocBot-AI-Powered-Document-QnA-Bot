# DocBot - AI-Powered Document Q&A Bot 📄🤖

**DocBot** is an AI-powered chatbot that allows users to query information from large documents like PDFs. It leverages **HuggingFace Embeddings** and **FAISS** for efficient document retrieval and integrates **LLMs** for generating accurate, context-based answers.

## Features
- 🔍 **Document Retrieval**: Uses **FAISS vector store** for fast and efficient document search.
- 🧠 **AI-Powered Answers**: Utilizes **Groq LLM (deepseek-llm-67b-chat)** for generating context-aware responses.
- 📁 **Source Referencing**: Displays **page numbers** and a snippet from source documents for every response.
- 🌐 **Streamlit Interface**: Simple and interactive UI built with **Streamlit** for easy user interaction.

## Tech Stack
- **Python**
- **LangChain** for chaining LLM with vector retrieval.
- **HuggingFace** Sentence Transformers for embeddings.
- **FAISS** for vector similarity search.
- **Streamlit** for building the web interface.
- **Groq LLM API** for language generation.

## How It Works
1. Upload documents and convert them into **vector embeddings** using **HuggingFace models**.
2. User inputs a query in the **Streamlit** interface.
3. **FAISS** retrieves the most relevant document chunks.
4. The **LLM** generates an accurate response based on the retrieved content.
5. Display the answer along with **page numbers** and **document snippets** for transparency.

## Setup & Run Locally
```bash
git clone https://github.com/your-username/DocBot.git
cd DocBot
pip install pipenv --user
pipenv shell
pip install -r requirements.txt
streamlit run docbot.py
```

## Future Improvements
- Add support for **multiple document uploads**.
- Integrate **more LLM models** for flexibility.
- Deploy on **AWS** for broader access.

---

