import joblib
import time
import uuid
import pandas as pd
from database import insert_transaction, insert_prediction, init_db

CONFIDENCE_THRESHOLD = 0.85

def feature_vector(row):
    # Using all numeric PCA features as-is
    fv = row.drop(['Class']).values.astype(float).reshape(1, -1)
    return fv

def main():
    print('Initialize DB...')
    init_db()

    print('Load model...')
    model = joblib.load('fraud_model_v1.pkl')

    print('Load dataset...')
    data = pd.read_csv('D:\\11. Projects\\Kurukshetra\\2\\archive (2)\\creditcard.csv')

    # Simulate streaming 100 random transactions
    sample_data = data.sample(100, random_state=42)

    for idx, row in sample_data.iterrows():
        tid = str(uuid.uuid4())
        features = row.drop('Class').to_dict()
        insert_transaction(tid, features)

        fv = feature_vector(row)
        pred_prob = model.predict_proba(fv)[0][1]
        conf = max(model.predict_proba(fv)[0])
        label = 'fraud' if pred_prob > 0.5 else 'legitimate'

        pid = str(uuid.uuid4())
        insert_prediction(pid, tid, pred_prob, conf, label)

        print(f'Transaction {tid[:6]} - Fraud Prob: {pred_prob:.3f}, Confidence: {conf:.3f}, Label: {label}')
        
        if conf < CONFIDENCE_THRESHOLD:
            print(f'  -> Needs human review.')
        else:
            print(f'  -> Auto-approved.')
        time.sleep(1)

if __name__ == "__main__":
    main()
