import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain.schema import Document
import shutil

# Load environment variables
load_dotenv()

# Check if running in cloud environment
IS_CLOUD = os.getenv("STREAMLIT_SHARING", False) or os.getenv("RAILWAY_ENVIRONMENT", False)

if IS_CLOUD:
    # Use Groq for cloud deployment (free open-source models)
    from langchain_groq import ChatGroq
    LLM_TYPE = "groq"
else:
    # Use Ollama for local deployment
    from langchain_community.llms import Ollama
    LLM_TYPE = "ollama"

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local storage paths
CHROMA_DB_PATH = "./chroma_db"
DATA_PATH = "./data"

# Custom CSS (same as before)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e88e5;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #1e88e5;
    color: #333;
}
.bot-message {
    background-color: #f1f8e9;
    border-left: 4px solid #4caf50;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings model."""
    model_name = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def initialize_llm():
    """Initialize LLM based on environment."""
    if LLM_TYPE == "groq":
        # Cloud deployment with Groq (free open-source models)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Please set GROQ_API_KEY in your environment variables for cloud deployment.")
            st.stop()
        return ChatGroq(
            model="llama-3.1-8b-instant",  # Free Groq model
            groq_api_key=groq_api_key,
            temperature=0.1
        )
    else:
        # Local deployment with Ollama
        return Ollama(
            model="llama3.2:3b",
            base_url="http://localhost:11434"
        )

# ... (rest of your functions remain the same)

def main():
    st.markdown('<h1 class="main-header">🩺 Medical Chatbot Assistant</h1>', unsafe_allow_html=True)
    
    # Show deployment info
    if IS_CLOUD:
        st.info("🌐 Running in cloud mode with Groq (Free Open-Source Models)")
    else:
        st.info("🏠 Running in local mode with Ollama")
    
    # ... (rest of your main function)

if __name__ == "__main__":
    main()