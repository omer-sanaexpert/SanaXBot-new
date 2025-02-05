# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point explicitly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
