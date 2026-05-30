from fastapi import FastAPI

app = FastAPI(
    title="Durian HR RAG Chatbot"
)

@app.get("/")
def root():
    return {
        "message": "Durian RAG API Running"
    }