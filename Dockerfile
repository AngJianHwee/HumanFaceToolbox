# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the examples (optional, for demonstration)
# CMD ["python", "examples/canny_example.py"]
