import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from database import init_db
import sqlite3

MIN_FEEDBACK_FOR_RETRAIN = 20

def fetch_feedback():
    conn = sqlite3.connect('feedback.db')
    df = pd.read_sql_query('''
    SELECT t.features, f.corrected_label 
    FROM feedback f
    JOIN predictions p ON f.prediction_id = p.prediction_id
    JOIN transactions t ON p.transaction_id = t.transaction_id
    ''', conn)
    conn.close()
    return df

def main():
    init_db()
    df = fetch_feedback()

    if len(df) < MIN_FEEDBACK_FOR_RETRAIN:
        print(f"Need at least {MIN_FEEDBACK_FOR_RETRAIN} feedback samples to retrain. Currently have {len(df)}.")
        return
    
    print(f"{len(df)} feedback samples collected.")
    user_input = input("Do you want to retrain the model now? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Retraining skipped.")
        return
    
    # Prepare features and labels
    X = pd.DataFrame(df['features'].apply(eval).tolist())
    y = (df['corrected_label'] == 'fraud').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    print('Retraining model...')
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f'New model accuracy: {acc:.4f}')
    
    joblib.dump(model, 'fraud_model_v_updated.pkl')
    print('Updated model saved as fraud_model_v_updated.pkl')

if __name__ == '__main__':
    main()
