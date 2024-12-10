# Use the latest Ubuntu image
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /usr/app/src 

# Define an argument for the language environment, defaulting to US English
ARG LANG="eng_US.UTF-8"

# Install necessary packages and tools
# - Update the package list
# - Install required packages:
#   - apt-utils: Provides additional utilities for package management
#   - locales: Handles language and character encoding settings
#   - python3-pip: Python package installer
#   - python3-yaml: YAML parser and emitter for Python
#   - rsyslog: System logging utility
#   - systemd: System and service manager
#   - systemd-cron: Cron job scheduler compatible with systemd
#   - sudo: Allows specific users to run commands as the superuser
# - Clean up after installation to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    locales \
    python3-pip \
    python3-yaml \
    rsyslog \
    systemd \
    systemd-cron \
    sudo && \
    apt-get clean

# Copy the requirements.txt file to the current directory in the container
COPY requirements.txt ./

# Upgrade pip and install Python dependencies listed in requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy the current directory contents into the container's working directory
COPY ./ ./
