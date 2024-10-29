from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel  # Import BaseModel for data validation
from app.data_processing import extract_code_files, create_prompt_response_pairs
from app.model_handler import LLMService
from app.repo_handler import clone_repos, pull_repo

app = FastAPI()
llm_service = LLMService(model_name="gpt2")
git_urls = [
    "https://github.com/Kotak-Neo/kotak-neo-api.git",
    "https://github.com/iterativv/NostalgiaForInfinity.git",
    "https://github.com/freqtrade/freqtrade.git",
    "https://github.com/zerodha/pykiteconnect.git"
] # Add more repositories as needed

clone_dir = "repo"  # Specify your clone directory here


# Define a request model for the prompt
class CodeRequest(BaseModel):
    prompt: str


@app.on_event("startup")
def startup_event():
    # Clone the repository if needed (uncomment when ready)
    clone_repos(git_urls)
    code_snippets = extract_code_files()
    data_pairs = create_prompt_response_pairs(code_snippets)

    # Log the data pairs for debugging
    print(data_pairs)

    # Sample data for initial fine-tuning (optional)
    sample_data_pairs = [
        {"prompt": "Write a Python function to add two numbers:", "response": "def add(a, b):\n    return a + b"},
        {"prompt": "Create a function to calculate the factorial of a number:",
         "response": "def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)"},
        {"prompt": "Write a Python script to connect to a database:",
         "response": "import sqlite3\n\ndef connect_to_db(db_name):\n    conn = sqlite3.connect(db_name)\n    return conn"},
        {"prompt": "Write a Python function to reverse a string:",
         "response": "def reverse_string(s):\n    return s[::-1]"},
        {"prompt": "Create a function to check if a number is prime:",
         "response": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"}
    ]

    # Fine-tune the model with the initial data pairs
    llm_service.fine_tune(sample_data_pairs + data_pairs)


@app.post("/generate_code/")
async def generate_code(request: CodeRequest):  # Change to accept a CodeRequest model
    try:
        # Generate code using the prompt
        #generated_code = llm_service.generate_code(request.prompt)
        generated_code = llm_service.generate_code(request.prompt, temperature=0.8, top_k=40)

        return {"generated_code": generated_code}
    except Exception as e:
        # Raise an HTTP exception if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/")
async def github_webhook(request: Request):
    payload = await request.json()
    if payload.get("ref") == "refs/heads/main":
        # Pull the latest changes from the repo if needed (uncomment when ready)
        pull_repo()
        code_snippets = extract_code_files()
        data_pairs = create_prompt_response_pairs(code_snippets)

        # Fine-tune the model with the newly extracted data pairs
        llm_service.fine_tune(data_pairs)

    return {"status": "Repository updated and model retrained"}


# Add a root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "Welcome to the LLM Code Generation API!"}


# Run the application using Uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
