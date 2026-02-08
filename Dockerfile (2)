
# Use official Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and scripts into container
COPY model.joblib .
COPY preprocessor.joblib .
COPY serve_model.py .

# Expose port (optional for API)
EXPOSE 8080

# Command to run the API/server script
CMD ["python", "serve_model.py"]
