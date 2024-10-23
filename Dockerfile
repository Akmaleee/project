# Use Python 3.8 slim image
FROM python:3.8-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]
