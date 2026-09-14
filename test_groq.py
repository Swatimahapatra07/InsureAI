from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq

# Load variables from .env
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    exit()

print("Groq API key loaded successfully!")

# Create Groq model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Test the model
response = llm.invoke("What is health insurance? Explain in one sentence.")

print("\nGroq Response:")
print(response.content)