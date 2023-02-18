FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade /code/requirements.txt

COPY ./dog_classifier_app /code/dog_classifier_app
