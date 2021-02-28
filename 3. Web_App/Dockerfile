
FROM tiangolo/uwsgi-nginx-flask

RUN apt-get update
RUN python -m pip install --upgrade pip

COPY ./app /app

RUN pip install -r /app/requirements.txt