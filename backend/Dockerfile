FROM python:3.11.8-slim-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 80

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]