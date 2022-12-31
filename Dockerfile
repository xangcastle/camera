FROM python:3.8.10
LABEL maintainer="Cesar Rammirez @xangcastle"

EXPOSE 8501

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 freeglut3-dev libgtk2.0-dev  -y
RUN apt-get install wget cmake gcc g++ build-essential -y

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install streamlit
RUN pip install -r requirements.txt

COPY . .
COPY ./patch/builder.py /usr/local/lib/python3.8/site-packages/google/protobuf/internal/builder.py

ENTRYPOINT [ "streamlit", "run"]
CMD ["app.py"]