# KurukShetra_Winners

## Project Overview

KurukShetra_Winners is a Human-in-the-Loop (HITL) AI Feedback System focused on dynamic fraud detection. It combines advanced machine learning with human expertise to deliver high-accuracy, continuously improving predictions on transactional data.

## Repository Structure

- **Model/**  
  Contains model training, prediction, retraining scripts, database interface, and pre-trained model artifacts.
  - `train.py`, `retrain_model.py`, `predict_and_stream.py`: Model training and inference workflows.
  - `database.py`, `feedback.db`: Feedback data management and storage.
  - `fraud_model_v1.pkl`: Saved hybrid model.
  - `cli_review.py`: Command-line reviewer utility.

- **UI/**  
  Holds frontend and backend code for user interaction.
  - `backend/`: FastAPI backend serving predictions and collecting feedback.
    - `main.py`: Entry point for the API.
    - `static/`: Frontend assets (`index.html`) for human review dashboard.
  - `README.md`: Project/documentation reference.

## Features

- Hybrid ML model: Deep Neural Networks, Autoencoder, and LSTM for robust fraud detection.
- Real-time transaction inference via Python API backend.
- Human review and feedback interface (web and CLI).
- Continuous feedback loop for model retraining and optimization.
- Centralized database for corrections and annotations.
- Ready for deployment (Vercel/Cloud/Container).

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database recommended for feedback storage.

### Backend Startup

```bash
cd UI/backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Usage

- Open `UI/backend/static/index.html` in your browser for the reviewer interface.
- Alternatively, serve via the backend for integrated access.

## Deployment

- Compatible with Vercel and other platforms supporting monorepo deployments.
- Recommend splitting frontend and backend as separate projects for scalable deployment.

## Contributing

Have ideas or improvements? Open an issue or submit a pull request!
