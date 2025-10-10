# 🩺 Medical Chatbot - RAG-Based AI Assistant

A sophisticated medical information assistant powered by Retrieval-Augmented Generation (RAG) technology, combining local language models with medical document knowledge base for accurate, contextual responses.

![Medical Chatbot](https://img.shields.io/badge/Medical-Chatbot-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-LLM-purple.svg)

## 🌟 Features

- **🤖 RAG-Powered Responses**: Combines document retrieval with AI generation for accurate medical information
- **🏠 Fully Local Operation**: No external API dependencies, complete privacy and control
- **⚡ Fast Performance**: Local vector database and LLM for quick responses
- **📚 Medical Knowledge Base**: Processes PDF medical documents for contextual answers
- **🔍 Semantic Search**: Advanced embeddings for relevant document retrieval
- **💬 Interactive UI**: Beautiful Streamlit web interface
- **🔧 Easy Management**: Built-in database management and system status monitoring
- **⚠️ Safety First**: Clear medical disclaimers and responsible AI practices

## 🛠️ Tech Stack

### **Backend**
- **🐍 Python 3.8+**: Core programming language
- **🦜 LangChain**: Framework for building LLM applications
- **🦙 Ollama**: Local LLM inference server
- **📊 ChromaDB**: Local vector database for embeddings storage
- **🤗 HuggingFace**: Embedding models (`BAAI/bge-small-en-v1.5`)
- **📄 PyPDF**: PDF document processing

### **Frontend Options**
- **🎨 Streamlit**: Interactive web application (Recommended)
- **🚀 FastAPI**: REST API with built-in web interface
- **💻 HTML/CSS/JS**: Custom web interface

### **AI Models**
- **Language Models**: 
  - **Local**: Llama 3.2 (3B/8B) via Ollama - Privacy-focused inference
  - **Cloud**: Llama 3.1 (8B) via Groq - Free open-source models
- **Embeddings**: BAAI/bge-small-en-v1.5 - Semantic understanding
- **Vector Store**: ChromaDB - Local similarity search

### **Deployment**
- **🐳 Docker**: Containerized deployment
- **☁️ Cloud Platforms**: Streamlit Cloud, Railway, Heroku
- **🔐 API Services**: Groq (Free), Ollama (Local)

## 📋 Prerequisites

- **Python 3.8 or higher**
- **Ollama installed**: [Installation Guide](https://ollama.ai/)
- **4GB+ RAM**: For running local LLM models
- **PDF Documents**: Medical documents for the knowledge base

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Godwill98/Compete-Medical-chatbot-Llms.git
cd Compete-Medical-chatbot-Llms
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv medibot
source medibot/bin/activate  # On Windows: medibot\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install and Set Up Ollama
```bash
# Install Ollama (visit https://ollama.ai for OS-specific instructions)

# Pull the required model
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

### 4. Prepare Your Medical Documents
```bash
# Create data directory and add your PDF files
mkdir data
# Copy your medical PDF files to the data/ directory
```

### 5. Create Local Vector Database
```bash
python create_local_db.py
```

### 6. Run the Application
```bash
# Option 1: Streamlit Web App (Recommended)
streamlit run streamlit_app.py

# Option 2: FastAPI Server
python fastapi_app.py

# Option 3: Cloud-Compatible App (for deployment)
streamlit run streamlit_app_cloud.py
```

## 🌐 Cloud Deployment

For cloud deployment, use `streamlit_app_cloud.py` which supports **Groq API** for free open-source models:

### **Environment Variables for Cloud**
```bash
# Required for cloud deployment
GROQ_API_KEY=your_groq_api_key_here

# Optional deployment flags (auto-detected)
STREAMLIT_SHARING=true
RAILWAY_ENVIRONMENT=true
```

### **Get Groq API Key (Free)**
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Generate your API key
4. Set it in your deployment platform's environment variables

### **Supported Cloud Platforms**
- **Streamlit Cloud**: Direct deployment from GitHub
- **Railway**: One-click deployment
- **Heroku**: Using Docker or buildpacks
- **Render**: Static site deployment

## 📁 Project Structure

```
Compete-Medical-chatbot-Llms/
├── 📄 README.md                 # Project documentation
├── 📋 requirements.txt          # Python dependencies
├── 🔧 setup.py                  # Package setup
├── 🌍 .env                      # Environment variables (optional)
├── 📚 data/                     # PDF medical documents
│   └── Medical_book.pdf
├── 🗃️ chroma_db/               # Local vector database
├── 🔬 research/                 # Jupyter notebooks for experimentation
│   └── trials.ipynb
├── 📦 src/                      # Source code modules
│   ├── helper.py
│   └── prompt.py
├── 🎨 streamlit_app.py          # Streamlit web application
├── 🚀 fastapi_app.py            # FastAPI server application
├── 🔨 create_local_db.py        # Vector database creation script
├── 📱 app.py                    # Alternative Flask application
└── 🤖 medibot/                  # Virtual environment
```

## 🎮 Usage

### **Streamlit Interface**
1. Open your browser to `http://localhost:8501`
2. Use the sidebar to:
   - Check system status
   - Test connections
   - Manage database
   - Clear chat history
3. Type medical questions in the chat input
4. Get AI-powered responses based on your documents

### **FastAPI Interface**
1. Open your browser to `http://localhost:8000`
2. Use the web interface for chat
3. Access API documentation at `http://localhost:8000/docs`

### **Sample Questions**
- "What is diabetes and its symptoms?"
- "How is hypertension treated?"
- "What are the side effects of aspirin?"
- "Explain the diagnosis process for pneumonia"

## ⚙️ Configuration

### **Model Selection**
Edit the model in your app files:
```python
# For different Ollama models
model="llama3.1:8b"    # Larger model, better quality
model="llama3.2:3b"    # Smaller model, faster inference
model="mistral:7b"     # Alternative model
```

### **Embedding Models**
```python
# Different embedding models
model_name = "BAAI/bge-small-en-v1.5"    # Default (384 dimensions)
model_name = "all-MiniLM-L6-v2"          # Alternative (384 dimensions)
```

### **Vector Database Settings**
```python
# Retrieval settings
search_kwargs={"k": 3}        # Number of documents to retrieve
chunk_size=500               # Document chunk size
chunk_overlap=20             # Overlap between chunks
```

## 🔧 Advanced Features

### **Database Management**
- **Rebuild Database**: Add new PDFs and recreate the vector store
- **Status Monitoring**: Check system health and connections
- **Cache Management**: Clear cached resources for fresh starts

### **API Endpoints (FastAPI)**
- `POST /chat`: Send messages to the chatbot
- `GET /health`: Check system health
- `GET /`: Web interface

## 🚨 Important Disclaimers

⚠️ **This chatbot is for informational purposes only**
- Always consult healthcare professionals for medical advice
- Not intended for emergency medical situations
- Responses based on training data, may not reflect latest research
- Should not replace professional medical diagnosis or treatment

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Community** for the excellent framework
- **Ollama Team** for local LLM inference
- **HuggingFace** for embedding models
- **ChromaDB** for vector database solution
- **Streamlit** for the amazing web app framework

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/Godwill98/Compete-Medical-chatbot-Llms/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Godwill98/Compete-Medical-chatbot-Llms/discussions)
- 📧 **Contact**: [Your Email]

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice interaction capabilities
- [ ] Integration with medical databases
- [ ] Advanced analytics and reporting
- [ ] Mobile application
- [ ] Doctor consultation scheduling
- [ ] Symptom checker integration

---

**Built with ❤️ by [Godwill Kiplagat](https://github.com/Godwill98)**

*Empowering healthcare through AI technology*