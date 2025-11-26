FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

ENV PORT=7860
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} app:server"]
