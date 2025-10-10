# 🚀 Deployment Guide

## Quick Deployment Options

### 1. 🐳 **Docker Deployment (Recommended)**

```bash
# Build and run with Docker Compose (easiest)
docker-compose up --build

# Or build manually
docker build -t medical-chatbot .
docker run -p 8501:8501 -p 11434:11434 -v ./data:/app/data medical-chatbot
```

**Access:** http://localhost:8501

### 2. ☁️ **Cloud Deployment**

#### **Streamlit Cloud**
1. Push code to GitHub
2. Visit https://share.streamlit.io/
3. Connect your repo
4. Set main file: `streamlit_app_cloud.py`
5. Add environment variable: `GROQ_API_KEY` (free at console.groq.com)

#### **Railway**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway add GROQ_API_KEY=your_groq_key
railway up
```

#### **Heroku**
```bash
# Install Heroku CLI
heroku create your-medical-chatbot
heroku config:set GROQ_API_KEY=your_groq_key
git push heroku main
```

### 3. 🖥️ **VPS Deployment**

```bash
# On your VPS
git clone https://github.com/Godwill98/Compete-Medical-chatbot-Llms.git
cd Compete-Medical-chatbot-Llms

# Install dependencies
pip install -r requirements.txt

# Install and setup Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3.2:3b

# Setup vector database
python create_local_db.py

# Run with PM2 (process manager)
npm install -g pm2
pm2 start "streamlit run streamlit_app.py --server.port 8501" --name medical-chatbot
```

### 4. 📱 **Mobile-Friendly Deployment**

For mobile access, use **ngrok** during development:

```bash
# Install ngrok
# Run your app locally
streamlit run streamlit_app.py

# In another terminal
ngrok http 8501
```

## 🔧 **Environment Variables**

Create `.env` file for sensitive data:

```env
# For cloud deployments (FREE with Groq)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Custom model settings
MODEL_NAME=llama3.2:3b
CHUNK_SIZE=500
```

### **Get Free Groq API Key**
1. Visit [console.groq.com](https://console.groq.com/)
2. Sign up for free account
3. Generate API key (no credit card required)
4. Free tier includes: Llama 3.1 8B, Mixtral 8x7B, and more!

## 📦 **Production Considerations**

### **Performance**
- Use Docker for consistent environments
- Consider GPU instances for faster inference
- Implement caching for frequently asked questions

### **Security**
- Use HTTPS in production
- Implement rate limiting
- Add user authentication if needed
- Keep API keys secure

### **Scaling**
- Use load balancers for multiple instances
- Consider cloud-managed vector databases
- Implement health checks and monitoring

### **Cost Optimization**
- Local deployment: Free (your hardware)
- Cloud with OpenAI: $0.002 per 1K tokens
- VPS: $5-50/month depending on specs

## 🏥 **Medical Compliance**

⚠️ **Important for production medical applications:**

- Add proper medical disclaimers
- Implement logging for audit trails  
- Consider HIPAA compliance if handling patient data
- Add content filtering for inappropriate medical advice
- Implement user consent mechanisms

## 🔍 **Monitoring & Maintenance**

```bash
# Health check endpoint (add to your app)
GET /health

# Logs monitoring
docker logs medical-chatbot
pm2 logs medical-chatbot

# Database backup
cp -r chroma_db chroma_db_backup_$(date +%Y%m%d)
```

## 🚀 **Quick Commands**

```bash
# Local development
streamlit run streamlit_app.py

# Docker development
docker-compose up

# Cloud deployment (Streamlit Cloud)
# Just push to GitHub and connect via web interface

# VPS deployment
git pull && docker-compose up --build -d
```

Choose the deployment method that best fits your needs:
- **Local/Testing**: Direct Python execution
- **Production**: Docker + VPS or Cloud platforms
- **Demo**: Streamlit Cloud (free)
- **Enterprise**: Custom cloud infrastructure