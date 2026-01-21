FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TI_DEVICE_MEMORY_GB=4

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxrandr2 \
    libxi6 \
    libxfixes3 \
    libxxf86vm1 \
    libxss1 \
    libegl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python -m pip install --upgrade pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

CMD ["python", "-c", "print('Container ready. Install dependencies and run the app.')"]
