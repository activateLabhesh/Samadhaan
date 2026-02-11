import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def list_models():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        models = client.models.list()
        print("Available Models:")
        for model in models.data:
            print(f"- {model.id}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_models()
