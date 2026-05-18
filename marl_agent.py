import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MARLEnv:
    """
    Multi‑agent environment where each ETF is an agent.
    State: recent returns of all agents (plus macro if desired)
    Action: 0 = hold, 1 = buy (or continuous weight)
    Reward: portfolio return based on actions
    """
    def __init__(self, returns_df, lookback=10):
        self.returns = returns_df.values  # (T, n_agents)
        self.n_agents = returns_df.shape[1]
        self.lookback = lookback
        self.current_step = lookback
        self.done = False

    def reset(self):
        self.current_step = self.lookback
        self.done = False
        return self._get_state()

    def _get_state(self):
        # state: (n_agents, lookback) – each agent's recent returns
        state = self.returns[self.current_step - self.lookback:self.current_step, :].T
        return torch.tensor(state, dtype=torch.float32)

    def step(self, actions):
        # actions: array of size (n_agents,) – 0 or 1 (buy)
        if self.current_step >= len(self.returns) - 1:
            self.done = True
            return self._get_state(), 0, self.done, {}
        # Compute reward: portfolio return of that day (using current actions)
        daily_returns = self.returns[self.current_step, :]  # actual returns
        # Portfolio return = mean of selected assets (equal weight among buys)
        if np.sum(actions) > 0:
            port_ret = np.mean(daily_returns[actions == 1])
        else:
            port_ret = 0.0
        self.current_step += 1
        next_state = self._get_state()
        return next_state, port_ret, self.done, {}

class QMixNet(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=64):
        super().__init__()
        self.agent_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                          nn.Linear(hidden_dim, 2)) for _ in range(n_agents)
        ])
        # Mixing network
        self.mix_net = nn.Sequential(nn.Linear(n_agents * hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))

    def forward(self, states):
        # states: (n_agents, state_dim)
        agent_qs = []
        for i, net in enumerate(self.agent_nets):
            q = net(states[i])
            agent_qs.append(q)
        return torch.stack(agent_qs)  # (n_agents, 2)

def train_marl(returns_df, window, n_episodes=100, lookback=10, lr=1e-3, gamma=0.99):
    """
    Train QMIX on the last `window` days of returns.
    Returns trained mixing network and agent networks.
    """
    env = MARLEnv(returns_df.iloc[-window:], lookback=lookback)
    n_agents = env.n_agents
    state_dim = lookback
    qmix = QMixNet(n_agents, state_dim)
    optimizer = optim.Adam(qmix.parameters(), lr=lr)
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                agent_qs = qmix(state)  # (n_agents, 2)
                actions = agent_qs.argmax(dim=1).numpy()
            next_state, reward, done, _ = env.step(actions)
            total_reward += reward
            state = next_state
        if (ep+1) % 20 == 0:
            print(f"    Episode {ep+1}/{n_episodes}, total reward: {total_reward:.4f}")
    return qmix, None, None

def get_weights(qmix, state):
    with torch.no_grad():
        agent_qs = qmix(state)
        buy_q = agent_qs[:, 1]
        return buy_q.numpy()
