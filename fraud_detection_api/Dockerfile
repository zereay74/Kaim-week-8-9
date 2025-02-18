# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the API
EXPOSE 5000

# Run the Flask API
CMD ["python", "serve_model.py"]

# Create Docker by the following command
# ' docker build -t fraud-detection-api . '

