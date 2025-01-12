from fastapi import FastAPI

def predict(query):
    if (query == "정말 기분 좋다"):
        return '기쁨'
    elif (query == "너무 짜증나"):
        return '화남'
# API Routing
app = FastAPI()
@app.get("/")
def read_root():
    return {"Let's Start!"}

@app.get("/model")
def read_root(query: str):
    return {"sentiment" : predict(query)}
