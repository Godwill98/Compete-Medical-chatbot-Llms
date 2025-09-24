#!/usr/bin/env python3
"""
Script to create local vector database from PDF documents
"""
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_local_vectorstore():
    """Create local vector database from PDF files in data directory."""
    
    # Paths
    DATA_PATH = "./data"
    CHROMA_DB_PATH = "./chroma_db"
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data directory '{DATA_PATH}' not found!")
        print("Please create the directory and add your PDF files.")
        return
    
    # Load documents
    print("📄 Loading PDF documents...")
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    
    if not documents:
        print(f"⚠️ No PDF files found in '{DATA_PATH}' directory.")
        return
    
    print(f"Found {len(documents)} documents")
    
    # Split documents
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} text chunks")
    
    # Initialize embeddings
    print("🔤 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Create vector store
    print("🗃️ Creating vector database...")
    if os.path.exists(CHROMA_DB_PATH):
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)
        print("Removed existing database")
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    # Persist the database
    vectorstore.persist()
    
    print(f"✅ Successfully created local vector database at '{CHROMA_DB_PATH}'")
    print(f"Database contains {len(texts)} document chunks")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    test_results = retriever.invoke("What is diabetes?")
    
    print(f"Retrieved {len(test_results)} relevant chunks for test query")
    if test_results:
        print(f"Sample result: {test_results[0].page_content[:200]}...")

if __name__ == "__main__":
    create_local_vectorstore()