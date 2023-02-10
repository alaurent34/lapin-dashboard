FROM python:3.10.8

ENV DASH_DEBUG_MODE False
COPY ./data /data
COPY ./assets /assets
COPY ./requirements.txt /requirements.txt
COPY ./lapin-0.2.0.tar.gz ./lapin-0.2.0.tar.gz
COPY ./app.py ./app.py 
WORKDIR /
RUN set -ex && \
    pip install -r requirements.txt &&\
    pip install ./lapin-0.2.0.tar.gz
EXPOSE 8050
CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server", "--timeout", "300"]