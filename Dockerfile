# Dockerfile
# Builds your RAG chatbot into a container

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first
# (Docker caches this layer — rebuilds are faster this way)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app when container starts
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]