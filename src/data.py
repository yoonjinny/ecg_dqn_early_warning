from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_demo_data(n=5000, length=187, abnormal_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    y = (rng.random(n) < abnormal_ratio).astype(int)
    X = []
    for label in y:
        base = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*11*t)
        noise = rng.normal(0, 0.08, length)
        x = base + noise
        if label == 1:
            center = rng.integers(30, 150)
            width = rng.integers(3, 10)
            amp = rng.uniform(1.0, 2.2)
            x[center:center+width] += amp
            x += 0.4*np.sin(2*np.pi*17*t)
        X.append(x)
    return np.array(X, dtype=np.float32), y.astype(np.int64)

def load_kaggle_csv(data_dir="data"):
    data_dir = Path(data_dir)
    candidates = list(data_dir.glob("*.csv"))
    if not candidates:
        return None, None
    csv_path = candidates[0]
    df = pd.read_csv(csv_path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values.astype(int)
    # Kaggle MIT-BIH labels: 0 normal, 1~4 arrhythmia classes. Binary conversion.
    y = (y_raw != 0).astype(np.int64)
    return X, y

def get_data(data_dir="data", test_size=0.2, seed=42, use_demo_if_missing=True):
    X, y = load_kaggle_csv(data_dir)
    if X is None:
        if not use_demo_if_missing:
            raise FileNotFoundError("Put Kaggle CSV files in ./data or enable demo data.")
        X, y = make_demo_data(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test
