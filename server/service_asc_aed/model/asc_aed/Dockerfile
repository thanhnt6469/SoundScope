#Dockerfile
FROM python:3.12.4
LABEL author="Lam-Pham-AIT"
WORKDIR /app
COPY . /app
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

CMD ["python", "asc_aed_detect.py"]


