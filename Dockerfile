FROM tensorflow/tensorflow:2.0.0

RUN mkdir /app
WORKDIR /app

RUN useradd -ms /bin/bash noneroot

RUN apt update
RUN apt install -y python3.7 python3-pip libpq-dev
RUN python3 -m  pip install --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install -r ./requirements.txt

USER noneroot
COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "valnet.wsgi"]