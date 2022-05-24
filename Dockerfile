FROM python:3.8.10

RUN pip install image
RUN pip install --upgrade tensorflow
COPY ecg_test_model.h5 /
ADD test/ test/
COPY main.py /

CMD [ "python", "./main.py" ]
