FROM python:3.11.5-slim

RUN apt-get update -q \
  && apt-get install --no-install-recommends -qy gcc inetutils-ping python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY /app .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
