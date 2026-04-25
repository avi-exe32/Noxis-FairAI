# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the cloud
WORKDIR /app

# Copy all your files into the cloud container
COPY . .

# Install your libraries
RUN pip install --no-cache-dir -r requirements.txt

# Create the uploads folder inside the cloud container so your app.py doesn't crash
RUN mkdir -p uploads

# Run the app using Gunicorn on the port Google Cloud provides
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 