FROM python:3.10-slim

WORKDIR /app

COPY requirements_api.txt .

RUN pip install --no-cache-dir -r requirements_api.txt

COPY app/ ./app/
COPY run_api.py .

RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]