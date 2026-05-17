import numpy as np
import torch

class MARLEnv:
    def __init__(self, returns, lookback=10):
        self.returns = returns.values      # (T, n_agents)
        self.n_agents = returns.shape[1]
        self.lookback = lookback
        self.t = lookback                  # start index
        self.max_steps = len(returns) - 1

    def reset(self):
        self.t = self.lookback
        state = self._get_state()
        return state

    def _get_state(self):
        # Each agent observes its own recent returns (lookback days)
        # State shape: (n_agents, lookback)
        state = self.returns[self.t-self.lookback:self.t, :].T
        return state

    def step(self, actions):
        # actions: (n_agents,) weights (not normalised)
        weights = np.exp(actions) / np.sum(np.exp(actions))  # softmax
        # portfolio return at time t
        rets = self.returns[self.t, :]
        portfolio_return = np.dot(weights, rets)
        # next state
        self.t += 1
        done = (self.t >= self.max_steps)
        next_state = self._get_state() if not done else None
        reward = portfolio_return
        return next_state, reward, done, {}
