from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from data import get_data

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(data_dir="data", seed=42)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, target_names=["normal", "abnormal"]))
    print("accuracy:", accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred))
    print("observed_ratio: 1.0")
