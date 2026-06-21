from fastapi import FastAPI

app = FastAPI()

@app.get("/")# route for root endpoint

def read_root():# function to handle GET request
    return {"Hello": "World"}

# to run this FastAPI application, use the command:
# uvicorn helloapi:app --reload

# --reload option allows the server to automatically reload when code changes are made

@app.get("/aboutsection")

def aboutsection():
    return {"About": "This is a simple FastAPI application that returns a greeting message."}