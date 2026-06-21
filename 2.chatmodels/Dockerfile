# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and source code
COPY req.txt .
COPY 2.chatmodel.py .

# Expose the port
EXPOSE 8000

# Install dependencies
RUN pip install --no-cache-dir -r req.txt

# Run the application
CMD ["python", "2.chatmodel.py"]