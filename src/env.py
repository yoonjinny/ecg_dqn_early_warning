import numpy as np

class ECGEarlyWarningEnv:
    """ECG early-warning environment.

    Action 0: continue observing
    Action 1: predict normal and stop
    Action 2: predict abnormal and stop
    """
    def __init__(self, X, y, step_size=10, max_len=187,
                 continue_penalty=-0.01, wrong_penalty=-2.0,
                 normal_reward=1.0, abnormal_reward=3.0):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.step_size = step_size
        self.max_len = max_len
        self.continue_penalty = continue_penalty
        self.wrong_penalty = wrong_penalty
        self.normal_reward = normal_reward
        self.abnormal_reward = abnormal_reward
        self.state_dim = max_len + 1
        self.rng = np.random.default_rng()
        self.idx = None
        self.t = None

    def reset(self, idx=None):
        self.idx = int(self.rng.integers(0, len(self.X))) if idx is None else int(idx)
        self.t = min(self.step_size, self.max_len)
        return self._state()

    def _state(self):
        s = np.zeros(self.state_dim, dtype=np.float32)
        s[:self.t] = self.X[self.idx, :self.t]
        s[-1] = self.t / self.max_len
        return s

    def step(self, action):
        y_true = int(self.y[self.idx])
        observed_ratio = self.t / self.max_len

        if action == 0:
            if self.t >= self.max_len:
                # Forced decision at the end: small penalty for delaying too long.
                return self._state(), -0.5, True, {"forced": True, "correct": False}
            self.t = min(self.t + self.step_size, self.max_len)
            return self._state(), self.continue_penalty, False, {"forced": False}

        pred = 0 if action == 1 else 1
        correct = pred == y_true
        if correct:
            base = self.abnormal_reward if y_true == 1 else self.normal_reward
            reward = base - observed_ratio
        else:
            reward = self.wrong_penalty
        return self._state(), float(reward), True, {
            "forced": False, "correct": correct, "pred": pred,
            "y_true": y_true, "observed_ratio": observed_ratio
        }
