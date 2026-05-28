#!/bin/bash

IMAGE_NAME="template" # Define the image name here so it's easy to change later

echo "Initiating teardown sequence for '$IMAGE_NAME'..."

echo "Checking for running or stopped containers..."
CONTAINERS=$(docker ps -a -q --filter ancestor="$IMAGE_NAME")

if [ -n "$CONTAINERS" ]; then
    echo "Destroying containers..."
    docker rm -f $CONTAINERS
else
    echo "No matching containers found."
fi

echo "Destroying the image..."
docker rmi -f "$IMAGE_NAME" 2>/dev/null || echo "Image '$IMAGE_NAME' not found."

echo "Teardown complete!"
