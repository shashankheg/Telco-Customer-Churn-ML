## Telco Churn – End-to-End ML Project
### Purpose

Build and ship a full machine-learning solution for predicting customer churn in a telecom setting—from data prep and modeling to an API + web UI deployed on Hugging face. 

### Problem solved & benefits

- Faster decisions: Predicts which customers are likely to churn so teams can act before they leave.
- Operationalized ML: Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- Repeatable delivery: CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- Traceable experiments: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST) and a root health check /.
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 7860
- CI/CD: GitHub Actions builds the image and pushes to Docker Hub deployed on Hugging face..
- Orchestration: Runs on the huggingface servers.


### Deployment flow (high-level)

- Push to main → GitHub Actions builds the Docker image and pushes it to Docker Hub.
- Users call POST /predict or open the Gradio UI at /ui via the ALB DNS.
- Use https://shashankheg-telco-customer-churn-ml.hf.space/ui/ to test the Ui
- For APi docs https://shashankheg-telco-customer-churn-ml.hf.space/docs

### Roadblocks & how we solved them

Buils issues due to dockerfile and Dockerfile.
>>Fix : The huggingface build users Dockerfile.

Module import error in container (ModuleNotFoundError: serving)

- Cause: Python path in the image didn’t include src/.
- Fixes: Set PYTHONPATH=/app/src in the Dockerfile; corrected uvicorn app path to src.app.main:app.


Gradio UI error (“No runs found in experiment”)

- Cause: Inference/UI expected an MLflow-logged model but couldn’t resolve a run.
- Fixes: Standardized MLflow experiment name and model logging in training; inference loads the logged model consistently (and a local path for dev).

Local testing vs. prod paths

- Cause: MLflow artifact URIs differ locally vs. in container.
- Fixes: For local dev, load via direct ./mlruns/.../artifacts/model; in prod, container loads the packaged model path used at build time.
