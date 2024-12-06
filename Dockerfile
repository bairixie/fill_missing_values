# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables to prevent Python from buffering output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Ensure model files are unzipped if they are present
RUN mkdir -p /app/model && \
    if [ -f /app/model/trained_bart_model.zip ]; then \
        unzip /app/model/trained_bart_model.zip -d /app/model && \
        rm /app/model/trained_bart_model.zip; \
    fi

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]