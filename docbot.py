import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("GROQ API key is missing! Please set the GROQ_API_KEY in your environment variables.")
        st.stop()
    
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=512
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Input prompt from user
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Check for casual greetings and respond accordingly
        if prompt.lower() in ['hi', 'hello', 'hey']:
            greeting_response = "Hello! How can I assist you today?"
            st.chat_message('assistant').markdown(greeting_response)
            st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
        else:
            CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer the user's question.
                If you don't know the answer, just say that you don't know. Don't try to make up an answer.
                Don't provide anything outside the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk, please.
            """

            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                # Get response from the QA chain
                response = qa_chain.invoke({'query': prompt})

                result = response["result"]
                source_documents = response["source_documents"]

                # Display result with better source information
                if source_documents:
                    source_info = "\n\n**Source Documents:**\n"
                    for i, doc in enumerate(source_documents):
                        page_number = doc.metadata.get('page', 'Unknown Page')
                        source_name = doc.metadata.get('source', 'Unknown Source')

                        # Extracting first 3-4 words from the paragraph
                        paragraph_start = ' '.join(doc.page_content.strip().split()[:4])

                        source_info += (
                            f"{i + 1}. **{source_name}** - *Page:* {page_number}\n"
                            f"*Starts with:* \"{paragraph_start}...\"\n\n"
                        )
                else:
                    source_info = "\n\n*No source documents found.*"

                result_to_show = result + source_info

                # Display assistant response
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
