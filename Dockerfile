FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p experiments models logs results dataset/imagenet_penalty

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.headless", "true", \
            "--server.port", "8501", \
            "--server.address", "0.0.0.0"]
