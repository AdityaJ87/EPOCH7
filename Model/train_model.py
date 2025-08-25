
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_initial_model(csv_path="D:\\11. Projects\\Kurukshetra\\2\\archive (2)\\creditcard.csv", model_path='fraud_model_v1.pkl'):
    print('Loading dataset...')
    data = pd.read_csv(csv_path)

    X = data.drop(['Class'], axis=1)
    y = data['Class']

    print('Splitting dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    print('Training RandomForestClassifier on dataset...')
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print('Saving trained model...')
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    train_initial_model()
