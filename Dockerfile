FROM python:3.11-slim

# working directory
WORKDIR /app

# install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# copy the entire project into the container
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# expose ports for both fastapi and streamlit
EXPOSE 8000
EXPOSE 8051


# CMD ["streamlit","run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# no command, because docker-compose tells each container what to run. 