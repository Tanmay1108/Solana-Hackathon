from fastapi import FastAPI

def get_app():
    app = FastAPI()
    return app

app = get_app()
# app.include_router(example.router)

# Root path
@app.get("/health")
def read_root():
    return {"message": "Welcome to Hack Solana!"}
