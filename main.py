from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def predict(query):
    if (query == "정말 기분 좋다"):
        return '기쁨'
    elif (query == "너무 짜증나"):
        return '화남'
    return ""

app = FastAPI()

# CORS middleware
origins = [
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routing
@app.get("/")
def read_root():
    return {"Let's Start!"}

@app.get("/model")
def read_root(query: str):
    return {"sentiment" : predict(query)}

