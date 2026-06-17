FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit plotly pandas sqlalchemy psycopg2-binary requests
COPY dashboard/ ./dashboard/
COPY app/ ./app/
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
