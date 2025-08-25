from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from typing import Dict, Any
from sqlalchemy import create_engine, Column, Integer, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow same origin and all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (including index.html)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# PostgreSQL database configuration
DATABASE_URL = "postgresql://fraud_user:strong_password@localhost:5432/fraud_detection"
try:
    engine = create_engine(DATABASE_URL)
    logger.info("Successfully connected to PostgreSQL database")
except Exception as e:
    logger.error(f"Failed to connect to database: {e}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(JSON)
    prediction = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class PredictionRequest(BaseModel):
    transaction_id: str
    data: Dict[str, Any]

class FeedbackRequest(BaseModel):
    transaction_id: str
    corrected_output: str

# Placeholder prediction logic
def predict_transaction(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"result": "Fraud" if data.get("TransactionAmt", 0) > 2000 else "Not Fraud", "confidence": 0.95, "details": data}

# API Endpoints
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request: {request}")
        prediction_result = predict_transaction(request.data)

        db = SessionLocal()
        db_prediction = Prediction(
            input_data=request.data,
            prediction=prediction_result
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        db.close()

        return {"transaction_id": request.transaction_id, **prediction_result}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    try:
        logger.info(f"Received feedback request: {request}")
        db = SessionLocal()
        db_prediction = db.query(Prediction).order_by(Prediction.id.desc()).first()
        if not db_prediction:
            db.close()
            raise HTTPException(status_code=404, detail="Prediction not found")

        db_prediction.timestamp = datetime.utcnow()
        db.commit()
        db.close()

        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        db.close()
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/history")
async def get_history():
    try:
        logger.info("Fetching history")
        db = SessionLocal()
        history = db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
        db.close()

        return [
            {
                "id": item.id,
                "input_data": item.input_data,
                "prediction": item.prediction,
                "timestamp": item.timestamp.isoformat()
            }
            for item in history
        ]
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        content = await file.read()
        data = json.loads(content)

        if not isinstance(data, dict) or "transaction_id" not in data:
            raise HTTPException(status_code=400, detail="Invalid JSON: transaction_id required")

        prediction_result = predict_transaction(data)

        db = SessionLocal()
        db_prediction = Prediction(
            input_data=data,
            prediction=prediction_result
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        db.close()

        return {"transaction_id": data["transaction_id"], **prediction_result}
    except json.JSONDecodeError:
        logger.error("Invalid JSON file")
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)