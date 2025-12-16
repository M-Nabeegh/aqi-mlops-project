# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first (to optimize build speed)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose port 8000 so we can access the API
EXPOSE 8000

# Run the API
# Matches your successful local command: "api.main:app"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]