import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_csv('phishing.csv')
    # Drop index column if present
    if 'Index' in df.columns:
        df = df.drop(columns=['Index'])

    # Features and label
    if 'class' not in df.columns:
        raise RuntimeError("Expected a 'class' column in phishing.csv")

    X = df.drop(columns=['class'])
    y = df['class']

    # simple train/test split for a quick sanity check
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained GradientBoostingClassifier, test accuracy: {acc:.4f}")

    # Save model
    with open('pickle/model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Saved model to pickle/model.pkl")

if __name__ == '__main__':
    main()
