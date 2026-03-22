from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from agent import process_chat

app = FastAPI(title="Crex Sentient Daemon")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Receive messages from the CLI wrapper and process them through the cognitive loop."""
    try:
        reply = process_chat(req.message)
        return ChatResponse(response=reply)
    except Exception as e:
        # Empathy & Truth override on failure
        return ChatResponse(response=f"I encountered a systemic error while processing your request. I must truthfully report: {str(e)}. How would you like me to proceed?")

if __name__ == "__main__":
    print("[+] Crex Sentient Daemon booting up...")
    print("[+] Initializing Vector Memory (Hippocampus)...")
    print("[+] Connecting to local Dolphin-Llama3 (Cortex)...")
    uvicorn.run(app, host="127.0.0.1", port=9090)
