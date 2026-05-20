# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app



# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# 5. Copy the entire project into the image
COPY /src ./src
COPY scripts ./scripts
COPY models_export/model /app/model


# Make "serving" and "app" importable without the "src." prefix
# # ensures logs are shown in real-time (no buffering).
# # lets you import modules using from app... instead of from src.app....
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/model

# 6. Expose FastAPI port
EXPOSE 7860


# 7. Run the FastAPI app using uvicorn (change path if needed)

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860"]