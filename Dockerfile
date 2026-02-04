FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

RUN apt update -y && apt upgrade -y
RUN apt install -y curl git wget vim python3 build-essential
RUN apt install -y openjdk-25-jdk

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN . $HOME/.local/bin/env
