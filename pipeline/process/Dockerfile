FROM python:3.7

WORKDIR /opt

ADD requirements.txt .

RUN pip install -r requirements.txt

ENV GOOGLE_APPLICATION_CREDENTIALS=hamoye-296618-8e48c5c60c4f.json

ADD . .

CMD ["python", "process.py", "--noauth_local_webserver"]