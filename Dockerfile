FROM python:3.7

WORKDIR /code

COPY ./ /code

ENV MODEL_DIR=/path/to/checkpoint
ENV MAX_LEN=60
ENV OS=10

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
