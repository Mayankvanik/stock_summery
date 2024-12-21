FROM python:3.10.12

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python","app_html_main.py"]