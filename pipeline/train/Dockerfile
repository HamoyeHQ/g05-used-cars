FROM python:3.7-stretch

WORKDIR /opt
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

ENTRYPOINT [ "python" ]
CMD ["train.py"]