FROM ubuntu:latest
RUN apt-get update && apt-get install python3-pip -y
ADD main.py main.py
CMD python3 main.py
