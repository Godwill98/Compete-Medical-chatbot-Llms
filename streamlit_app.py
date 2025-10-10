import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain.schema import Document
import shutil

# Load environment variables (optional now since we're not using Pinecone)
# If using Pinecone vector DB (Cloud-based)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local storage paths (Chroma DB Vector DB)
CHROMA_DB_PATH = "./chroma_db"
DATA_PATH = "./data"

# Custom CSS
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

def load_documents():
    """Load and process PDF documents."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data directory '{DATA_PATH}' not found. Please create it and add your PDF files.")
        return []
    
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    if not documents:
        st.warning(f"No PDF files found in '{DATA_PATH}' directory.")
        return []
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts = text_splitter.split_documents(documents)
    
    return texts

@st.cache_resource
def initialize_vector_store():
    """Initialize local Chroma vector store."""
    embeddings = initialize_embeddings()
    
    # Check if vector store already exists
    if os.path.exists(CHROMA_DB_PATH):
        st.info("Loading existing local vector database...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    else:
        st.info("Creating new local vector database...")
        # Load documents
        documents = load_documents()
        
        if not documents:
            st.error("No documents to process!")
            st.stop()
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        
        # Persist the vector store
        vectorstore.persist()
        st.success(f"Created local vector database with {len(documents)} document chunks!")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    return retriever

@st.cache_resource
def initialize_llm():
    """Initialize Ollama LLM."""
    return Ollama(
        model="llama3.2:3b",
        base_url="http://localhost:11434"
    )

@st.cache_resource
def initialize_rag_chain():
    """Initialize the RAG chain."""
    retriever = initialize_vector_store()
    llm = initialize_llm()
    
    system_prompt = (
        "You are a medical assistant that helps users find information about medical conditions based on a set of documents. "
        "Use the provided context to answer the user's question accurately and concisely. "
        "Otherwise, provide the best possible answer based on the context with the user request."
        "Use four sentences maximum."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def main():
    st.markdown('<h1 class="main-header">🩺 Medical Chatbot Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This medical chatbot uses:
        - **RAG (Retrieval-Augmented Generation)**
        - **Local Chroma DB** for vector storage
        - **Ollama** for local LLM
        - **Medical documents** as knowledge base
        """)
        
        st.header("📚 Database Management")
        
        # Show database status
        if os.path.exists(CHROMA_DB_PATH):
            st.success("✅ Local vector database found")
        else:
            st.warning("⚠️ No local database found")
            
        # Rebuild database button
        if st.button("🔄 Rebuild Database"):
            if os.path.exists(CHROMA_DB_PATH):
                shutil.rmtree(CHROMA_DB_PATH)
            st.cache_resource.clear()
            st.rerun()
            
        # Clear cache button
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared!")
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This chatbot is for informational purposes only. 
        Always consult with healthcare professionals for medical advice.
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
            
        st.header("🔧 System Status")
        if st.button("Test Connection"):
            with st.spinner("Testing Ollama connection..."):
                try:
                    llm = initialize_llm()
                    test_response = llm.invoke("Hello")
                    st.success("✅ Ollama connection working!")
                    st.info(f"Test response: {test_response[:100]}...")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Medical Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_question = st.chat_input("Ask me about medical conditions...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_question}</div>', unsafe_allow_html=True)
        
        # Get response from RAG chain
        with st.spinner("Searching medical knowledge base..."):
            try:
                rag_chain = initialize_rag_chain()
                response = rag_chain.invoke({"input": user_question})
                bot_response = response["answer"]
                
                # Add bot response to chat history
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                # Display bot response
                st.markdown(f'<div class="chat-message bot-message"><strong>Medical Assistant:</strong> {bot_response}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()