import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from env import ECGEarlyWarningEnv
from dqn import DQNAgent, ReplayBuffer
from data import get_data

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def train(episodes=3000, step_size=10, data_dir="data"):
    X_train, X_test, y_train, y_test = get_data(data_dir=data_dir, seed=SEED)
    env = ECGEarlyWarningEnv(X_train, y_train, step_size=step_size, max_len=X_train.shape[1])
    agent = DQNAgent(env.state_dim)
    replay = ReplayBuffer()

    eps_start, eps_end, eps_decay = 1.0, 0.05, 1200
    rewards, losses = [], []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay)
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)
            replay.push(state, action, reward, next_state, done)
            loss = agent.update(replay, batch_size=64)
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        if ep % 100 == 0:
            agent.sync_target()
        if (ep + 1) % 500 == 0:
            print(f"episode={ep+1}, avg_reward={np.mean(rewards[-100:]):.3f}, epsilon={epsilon:.3f}")

    torch.save(agent.q.state_dict(), "results/dqn_ecg.pt")
    plt.figure(); plt.plot(rewards); plt.title("DQN Training Reward"); plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.savefig("results/reward_curve.png", bbox_inches="tight")
    return agent, (X_test, y_test)

if __name__ == "__main__":
    train()
