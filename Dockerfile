# HeatGuard AI - container image for AWS (ECR -> ECS Fargate)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# IMPORTANT: pin scikit-learn / numpy / pandas in requirements.txt to the SAME
# versions you trained the .pkl models with (run `pip freeze` in your venv).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# 2 workers x 4 threads; long timeout because the first request loads the model + CSV
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120"]
