FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# Install git (required for openenv install)
RUN apt-get update && apt-get install -y git

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

EXPOSE 7860

# CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]