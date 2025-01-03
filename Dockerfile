# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy application code
COPY ./app /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
