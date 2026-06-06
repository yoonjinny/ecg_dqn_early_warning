import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from env import ECGEarlyWarningEnv
from dqn import DQNAgent
from data import get_data

def evaluate(model_path="results/dqn_ecg.pt", data_dir="data", step_size=10):
    _, X_test, _, y_test = get_data(data_dir=data_dir, seed=42)
    env = ECGEarlyWarningEnv(X_test, y_test, step_size=step_size, max_len=X_test.shape[1])
    agent = DQNAgent(env.state_dim)
    agent.q.load_state_dict(torch.load(model_path, map_location=agent.device))
    preds, ratios = [], []
    for i in range(len(X_test)):
        s = env.reset(idx=i); done = False
        pred = 0; ratio = 1.0
        while not done:
            a = agent.act(s, epsilon=0.0)
            s, r, done, info = env.step(a)
            if done:
                pred = info.get("pred", 0)
                ratio = info.get("observed_ratio", 1.0)
        preds.append(pred); ratios.append(ratio)
    print(classification_report(y_test, preds, target_names=["normal", "abnormal"]))
    print("accuracy:", accuracy_score(y_test, preds))
    print("f1:", f1_score(y_test, preds))
    print("avg_observed_ratio:", float(np.mean(ratios)))
    cm = confusion_matrix(y_test, preds)
    plt.figure(); plt.imshow(cm); plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm): plt.text(j, i, str(v), ha="center", va="center")
    plt.savefig("results/confusion_matrix.png", bbox_inches="tight")

if __name__ == "__main__":
    evaluate()
