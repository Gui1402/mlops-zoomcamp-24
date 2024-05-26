# mlflow/Dockerfile
FROM python:3.9-slim

RUN pip install mlflow psycopg2-binary boto3

ENTRYPOINT ["mlflow", "server"]
CMD ["--backend-store-uri", "postgresql://user:password@postgres:5432/mlflowdb", "--artifacts-destination", "s3://bucket", "--host", "0.0.0.0", "--port", "5000"]
