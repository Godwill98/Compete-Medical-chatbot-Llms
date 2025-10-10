"""
Test script to verify Groq integration works properly
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test Groq API connection and basic functionality"""
    
    # Check if API key is set
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        print("Please set your Groq API key in .env file")
        return False
    
    print("✅ Found Groq API key")
    
    try:
        # Initialize Groq Chat model
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.1
        )
        
        print("✅ Groq model initialized successfully")
        
        # Test basic query
        response = llm.invoke("Hello! Can you briefly explain what diabetes is?")
        print("✅ Groq API call successful")
        print(f"📝 Response: {response.content[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Groq connection: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Groq Integration...")
    print("=" * 50)
    
    success = test_groq_connection()
    
    print("=" * 50)
    if success:
        print("🎉 Groq integration test passed!")
        print("You can now deploy your app to the cloud with Groq")
    else:
        print("❌ Groq integration test failed")
        print("Please check your API key and try again")