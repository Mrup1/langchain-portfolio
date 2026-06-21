FROM python:3.9

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r Requirements.txt

# Expose both ports
EXPOSE 8000
EXPOSE 8501

# Run both FastAPI and Streamlit
CMD ["bash", "-c", "uvicorn app:app --port 8000 & sleep 5 && streamlit run streamlit.py --server.port=8501"]
