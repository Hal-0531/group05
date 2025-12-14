FROM python:3.12-slim-bookworm

# Environment variables
ENV UV_SYSTEM_PYTHON=1
ENV UV_PROJECT_ENVIRONMENT=.venv
ENV PATH="/root/work/.venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      wget \
      vim \
      ca-certificates \
      openssh-client \
      tmux \
      byobu \
      && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Set the working directory
WORKDIR /root/work
