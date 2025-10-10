#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait a bit for Ollama to start
sleep 5

# Pull the required model if not already available
ollama pull llama3.2:3b

# Create vector database if data directory has files
if [ -n "$(ls -A /app/data 2>/dev/null)" ]; then
    echo "Found PDF files, creating vector database..."
    python create_local_db.py
fi

# Start Streamlit
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0