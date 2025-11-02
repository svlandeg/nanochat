FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3.10-venv curl && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

COPY . /app/

CMD ["bash", "speedrun.sh"]