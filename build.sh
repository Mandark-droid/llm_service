
# Build the Docker image
docker build -t llm_service .

# Run the container
docker run -p 8000:8000 llm_service

## The API should now be accessible at http://localhost:8000.
### Non-Docker start
### uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
