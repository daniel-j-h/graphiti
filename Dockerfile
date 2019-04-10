FROM nvidia/cuda:10.1-runtime

WORKDIR /app
COPY . /app

RUN apt-get update -qq                                                     && \
    apt-get install -qq -y -o quiet=1 python3 python3-pip                  && \
    python3 -m pip install -r requirements-lock.txt

CMD ["python3", "-m", "graphiti"]
