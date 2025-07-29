FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install required libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY handler.py .

# Run the handler
CMD ["python", "handler.py"]
