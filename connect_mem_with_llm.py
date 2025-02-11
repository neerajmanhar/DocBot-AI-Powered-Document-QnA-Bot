import os
from openai import OpenAI  # For Groq API
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm_groq():
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", 
        api_key=GROQ_API_KEY
    )

    def query_groq(prompt):
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()

    return query_groq

# Step 2: Connect LLM with FAISS and Create Chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
groq_llm = load_llm_groq()

def qa_pipeline(query):
    # Retrieve documents
    retrieved_docs = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Apply custom prompt
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE).format(context=context, question=query)

    # Get answer from Groq
    answer = groq_llm(prompt)

    return {
        "result": answer,
        "source_documents": retrieved_docs
    }

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_pipeline(user_query)
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])
