# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code and model into the container
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Run the app using Gunicorn (a production web server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
