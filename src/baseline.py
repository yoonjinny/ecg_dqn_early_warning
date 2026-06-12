from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from data import get_data

'''
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(data_dir="data", seed=42)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred, target_names=["normal", "abnormal"]))
    print("accuracy:", accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred))
    print("observed_ratio: 1.0")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from data import get_data

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(data_dir="data", seed=42)

    # LSTM 입력 형태: (batch, seq_len, feature_dim)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=256,
        shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMBaseline().to(device)

    # class imbalance 보정
    class_count = torch.bincount(y_train)
    class_weight = len(y_train) / (2.0 * class_count.float())
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch={epoch}, loss={total_loss / len(train_loader):.4f}")

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            xb = X_test[i:i+512].to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu()
            preds.append(pred)

    pred = torch.cat(preds).numpy()

    print(classification_report(y_test, pred, target_names=["normal", "abnormal"]))
    print("accuracy:", accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred))
    print("observed_ratio: 1.0")

    torch.save(model.state_dict(), "results/lstm_baseline.pt")


'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from data import get_data

class MLPBaseline(nn.Module):
    def __init__(self, input_dim=187):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(data_dir="data", seed=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=256,
        shuffle=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPBaseline(input_dim=X_train.shape[1]).to(device)

    class_count = torch.bincount(y_train)
    class_weight = len(y_train) / (2.0 * class_count.float())
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"epoch={epoch}, loss={total_loss / len(train_loader):.4f}")

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), 512):
            xb = X_test[i:i+512].to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu()
            preds.append(pred)

    pred = torch.cat(preds).numpy()

    print(classification_report(y_test, pred, target_names=["normal", "abnormal"]))
    print("accuracy:", accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred))
    print("observed_ratio: 1.0")

    torch.save(model.state_dict(), "results/mlp_baseline.pt")
