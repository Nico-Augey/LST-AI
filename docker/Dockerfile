FROM tensorflow/tensorflow:2.15.0.post1-gpu

# Prevents prompts from asking for user input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libeigen3-dev \
    unzip \
    software-properties-common \
    curl

# Install Python 3.10 without using add-apt-repository
# RUN apt-get update && apt-get install -y python3.10 python3.10-distutils

# # Install pip for Python 3.10
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# # Upgrade requests and create a symlink for Python
# RUN pip3 install --upgrade requests && \
#     ln -fs /usr/bin/python3.10 /usr/bin/python

# Display Python version to verify installation
RUN python --version

# Setup LST-AI
RUN mkdir -p /custom_apps/lst_directory

# Installation de HD-BET
WORKDIR /custom_apps/lst_directory
RUN git clone https://github.com/MIC-DKFZ/HD-BET
WORKDIR /custom_apps/lst_directory/HD-BET
RUN pip install -e .

# Retrieve model weights for HD-BET
WORKDIR /custom_apps/lst_directory/
RUN mkdir -p /root/hd-bet_params
RUN wget -O /root/hd-bet_params/data.zip \
    https://zenodo.org/api/records/2540695/files-archive
WORKDIR /root/hd-bet_params/
RUN unzip data.zip && rm data.zip

# Clone and configuration of LST-AI
WORKDIR /custom_apps/lst_directory
RUN git clone https://github.com/Nico-Augey/LST-AI.git
WORKDIR /custom_apps/lst_directory/LST-AI
RUN git pull origin main

# Installation des dépendances pour LST-AI
RUN pip install -e .

# Retrieve model weights and files for LST-AI
WORKDIR /custom_apps/lst_directory/
RUN wget -O /custom_apps/lst_directory/LST-AI/LST_AI/data.zip \
    https://github.com/CompImg/LST-AI/releases/download/v1.1.0/lst_data.zip
WORKDIR /custom_apps/lst_directory/LST-AI/LST_AI/
RUN unzip data.zip && rm data.zip

# Download and configuration of greedy
WORKDIR /custom_apps/lst_directory
RUN wget "https://github.com/CompImg/LST-AI/releases/download/v1.0.0/greedy"
RUN chmod +x greedy
RUN mv greedy /usr/bin

# Make directories for easily mounting data
# You may change these to your liking
RUN mkdir -p /custom_apps/lst_input
RUN mkdir -p /custom_apps/lst_output
RUN mkdir -p /custom_apps/lst_temp

# Entrypoint to run the python script when the container starts
ENTRYPOINT [ "lst" ]
