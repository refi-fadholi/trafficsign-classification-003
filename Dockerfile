FROM python:3.8 AS python-base
ENV ENV=production \
    PORT=8001
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 8001
# Perintah untuk menjalankan FastAPI dengan Uvicorn
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# ENTRYPOINT ["sh", "-c", "/usr/local/bin/uvicorn main:app --host 0.0.0.0 --port $PORT"]