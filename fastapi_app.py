from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from typing import List, Optional
from langchain.schema import Document
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Medical Chatbot API",
    description="RAG-based Medical Assistant API using Ollama and Pinecone",
    version="1.0.0"
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    
class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Global variables for caching
embeddings = None
retriever = None
rag_chain = None

def initialize_embeddings():
    """Initialize embeddings model."""
    global embeddings
    if embeddings is None:
        model_name = "BAAI/bge-small-en-v1.5"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def initialize_pinecone():
    """Initialize Pinecone connection."""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Pinecone API key not found")
    return Pinecone(api_key=PINECONE_API_KEY)

def initialize_vector_store():
    """Initialize vector store and retriever."""
    global retriever
    if retriever is None:
        embeddings = initialize_embeddings()
        pc = initialize_pinecone()
        
        index_name = "medical-chatbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

def initialize_llm():
    """Initialize Ollama LLM."""
    return Ollama(
        model="llama3.2:3b",  # Using smaller model for memory efficiency
        base_url="http://localhost:11434"
    )

def initialize_rag_chain():
    """Initialize the RAG chain."""
    global rag_chain
    if rag_chain is None:
        retriever = initialize_vector_store()
        llm = initialize_llm()
        
        system_prompt = (
            "You are a medical assistant that helps users find information about medical conditions based on a set of documents. "
            "Use the provided context to answer the user's question accurately and concisely. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum."
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

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        initialize_rag_chain()
        print("✅ RAG chain initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG chain: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Serve the frontend HTML."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Chatbot</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { background: #1e88e5; color: white; padding: 20px; text-align: center; border-radius: 10px; margin-bottom: 20px; }
            .chat-container { background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); height: 500px; display: flex; flex-direction: column; }
            .messages { flex: 1; overflow-y: auto; padding: 20px; }
            .message { margin: 10px 0; padding: 10px; border-radius: 10px; max-width: 80%; }
            .user-message { background: #e3f2fd; margin-left: auto; text-align: right; }
            .bot-message { background: #f1f8e9; }
            .input-container { padding: 20px; border-top: 1px solid #eee; display: flex; gap: 10px; }
            .input-field { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            .send-button { padding: 10px 20px; background: #1e88e5; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .send-button:hover { background: #1976d2; }
            .disclaimer { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🩺 Medical Chatbot Assistant</h1>
                <p>Powered by RAG, Ollama & Pinecone</p>
            </div>
            
            <div class="disclaimer">
                <strong>⚠️ Disclaimer:</strong> This chatbot is for informational purposes only. Always consult with healthcare professionals for medical advice.
            </div>
            
            <div class="chat-container">
                <div class="messages" id="messages"></div>
                <div class="input-container">
                    <input type="text" class="input-field" id="messageInput" placeholder="Ask me about medical conditions..." onkeypress="handleKeyPress(event)">
                    <button class="send-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>

        <script>
            const messagesContainer = document.getElementById('messages');
            const messageInput = document.getElementById('messageInput');

            function addMessage(content, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = isUser ? 'message user-message' : 'message bot-message';
                messageDiv.innerHTML = `<strong>${isUser ? 'You' : 'Medical Assistant'}:</strong> ${content}`;
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }

            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;

                addMessage(message, true);
                messageInput.value = '';
                
                addMessage('Thinking...', false);
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message })
                    });
                    
                    const data = await response.json();
                    
                    // Remove "Thinking..." message
                    messagesContainer.removeChild(messagesContainer.lastChild);
                    
                    if (response.ok) {
                        addMessage(data.response, false);
                    } else {
                        addMessage('Sorry, I encountered an error: ' + data.detail, false);
                    }
                } catch (error) {
                    messagesContainer.removeChild(messagesContainer.lastChild);
                    addMessage('Sorry, I encountered a connection error.', false);
                }
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }

            // Add welcome message
            addMessage('Hello! I\'m your medical assistant. Ask me about medical conditions, symptoms, or treatments.', false);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test if Ollama is accessible
        llm = initialize_llm()
        return HealthResponse(status="healthy", message="All systems operational")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for medical queries."""
    try:
        rag_chain = initialize_rag_chain()
        response = rag_chain.invoke({"input": request.message})
        
        # Extract sources if available
        sources = []
        if "source_documents" in response:
            sources = [doc.metadata.get("source", "Unknown") for doc in response["source_documents"]]
        
        return ChatResponse(
            response=response["answer"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)