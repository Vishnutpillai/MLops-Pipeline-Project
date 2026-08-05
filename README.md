# 🚀 Production-Grade Insurance Cost Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-0194E2?logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)
![Feast](https://img.shields.io/badge/Feast-Feature%20Store-4B8BBE)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green)

> End-to-end production-grade MLOps pipeline for Insurance Cost
> Prediction with DVC, Feast, MLflow, FastAPI, Docker, Prometheus,
> Grafana, and GitHub Actions.

## 👥 Project Team

  ----------------------------------------------------------------------------------------------------------
  Member                LinkedIn                                          GitHub
  --------------------- ------------------------------------------------- ----------------------------------
  **Vishnu T Pillai**   https://www.linkedin.com/in/vishnu-t-pillai       https://github.com/Vishnutpillai

  **Sreyas**            https://www.linkedin.com/in/sreyas-s-582b61364/   https://github.com/Sreyas2255

  **Harani**            https://www.linkedin.com/in/harani-s/             https://github.com/Harani-S1
  ----------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📌 Project Overview

This project demonstrates a complete end-to-end Machine Learning
Operations (MLOps) workflow, covering:

-   Data Ingestion
-   Data Engineering
-   Feature Engineering
-   Feature Store (Feast)
-   Model Training
-   Model Evaluation
-   Model Registry (MLflow)
-   Deployment (FastAPI + Docker)
-   Monitoring (Prometheus + Grafana)
-   CI/CD Automation (GitHub Actions)

## 🛠️ Technology Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   DVC
-   Feast
-   MLflow
-   FastAPI
-   Docker
-   Prometheus
-   Grafana
-   GitHub Actions

## 📁 Project Structure

``` text
app/
src/
data/
feature_store/
models/
monitoring/
evaluation/
.github/workflows/
Dockerfile
requirements.txt
dvc.yaml
prometheus.yml
```

## ▶️ Run the Project

``` bash
pip install -r requirements.txt
uvicorn app.app:app --reload
```

FastAPI Docs: - http://127.0.0.1:8000/docs

Prometheus: - http://localhost:9090

Grafana: - http://localhost:3000

## 📈 Monitoring

-   Prediction metrics
-   API request metrics
-   Latency monitoring
-   Prometheus scraping
-   Grafana dashboards
-   Email alert support

## 🔄 CI/CD

GitHub Actions automates build, testing, training, deployment, and
monitoring.

## 📄 License

MIT License.
