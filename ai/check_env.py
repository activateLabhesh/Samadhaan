import os
from dotenv import load_dotenv
from groq import Groq

# Load .env
load_dotenv()

def check_groq():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        print("❌ GROQ_API_KEY is missing!")
        return
    
    print(f"✅ GROQ_API_KEY found: {api_key[:4]}...{api_key[-4:]}")
    print(f"📡 Using Model: {model_name}")
    
    try:
        client = Groq(api_key=api_key)
        # Try a simple chat completion to verify key
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            model=model_name,
        )
        print("✅ Groq Chat API connection successful!")
        print(f"Response: {chat_completion.choices[0].message.content[:50]}...")
    except Exception as e:
        print(f"❌ Groq Chat API connection failed: {e}")

def check_transcription():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
    
    # Create a dummy silent wav file for testing
    import wave
    test_file = "test_silent.wav"
    with wave.open(test_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b'\x00' * 88200) # 1 second of silence
    
    try:
        client = Groq(api_key=api_key)
        with open(test_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(test_file, f.read()),
                model=model_name,
            )
        print(f"✅ Groq Transcription API successful with model: {model_name}!")
        print(f"Result: '{transcription.text}'")
    except Exception as e:
        print(f"❌ Groq Transcription API failed: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    check_groq()
    print("-" * 20)
    check_transcription()
