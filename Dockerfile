# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:latest-gpu

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
# TensorFlow is already included in the base image, so no need to install it separately
RUN pip install --no-cache-dir -r requirements.txt